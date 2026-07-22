#include "comm/UartMavlinkLink.hpp"

#include "mavlink/myformat/myformat.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace {

// termios cfsetispeed/cfsetospeed sabit B* değerleri bekler — MAVLink'te
// yaygın kullanılan hızlarla sınırlı tutuyoruz (CLAUDE.md'de baud
// belirtilmemiş, saha pratiğinde en yaygın varsayılan 57600).
speed_t toTermiosBaud(uint32_t baud) {
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        case 460800: return B460800;
        case 921600: return B921600;
        default: return B57600;
    }
}

}  // namespace

struct UartMavlinkLink::Impl {
    std::string device;
    uint32_t baud;
    BboxCallback bbox_cb;

    int fd = -1;
    std::thread read_thread;
    std::atomic<bool> running{false};

    mavlink::mavlink_message_t rx_msg{};
    mavlink::mavlink_status_t rx_status{};

    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> messages_parsed{0};
    std::atomic<uint64_t> bbox_received{0};
    std::atomic<uint64_t> parse_errors{0};

    Impl(std::string dev, uint32_t b) : device(std::move(dev)), baud(b) {}

    bool openPort() {
        fd = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            std::cerr << "[UartMavlinkLink] " << device
                      << " açılamadı: " << std::strerror(errno) << "\n";
            return false;
        }

        termios tty{};
        if (tcgetattr(fd, &tty) != 0) {
            std::cerr << "[UartMavlinkLink] tcgetattr başarısız: "
                      << std::strerror(errno) << "\n";
            ::close(fd);
            fd = -1;
            return false;
        }

        speed_t speed = toTermiosBaud(baud);
        cfsetispeed(&tty, speed);
        cfsetospeed(&tty, speed);

        // Raw 8N1 — MAVLink kendi framing'ini (STX/len/checksum) taşıyor,
        // satır sonu/parity/flow-control dönüşümleri veri bozar.
        cfmakeraw(&tty);
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 1;  // 100ms poll aralığı (O_NONBLOCK ile birlikte)

        if (tcsetattr(fd, TCSANOW, &tty) != 0) {
            std::cerr << "[UartMavlinkLink] tcsetattr başarısız: "
                      << std::strerror(errno) << "\n";
            ::close(fd);
            fd = -1;
            return false;
        }
        tcflush(fd, TCIOFLUSH);
        return true;
    }

    void readLoop() {
        uint8_t buf[256];
        while (running.load()) {
            ssize_t n = ::read(fd, buf, sizeof(buf));
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    continue;
                }
                std::cerr << "[UartMavlinkLink] read hatası: " << std::strerror(errno)
                          << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (n == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            bytes_read.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);

            for (ssize_t i = 0; i < n; ++i) {
                uint8_t result = mavlink::mavlink_parse_char(0, buf[i], &rx_msg, &rx_status);
                if (result != mavlink::MAVLINK_FRAMING_OK) {
                    continue;
                }
                messages_parsed.fetch_add(1, std::memory_order_relaxed);

                if (rx_msg.msgid == mavlink::myformat::msg::BBOX::MSG_ID) {
                    mavlink::myformat::msg::BBOX bbox_msg{};
                    mavlink::MsgMap map(&rx_msg);
                    bbox_msg.deserialize(map);

                    BBox bbox;
                    bbox.x = static_cast<int>(bbox_msg.x);
                    bbox.y = static_cast<int>(bbox_msg.y);
                    bbox.width = static_cast<int>(bbox_msg.w);
                    bbox.height = static_cast<int>(bbox_msg.h);

                    if (bbox.width <= 0 || bbox.height <= 0) {
                        parse_errors.fetch_add(1, std::memory_order_relaxed);
                        std::cerr << "[UartMavlinkLink] geçersiz BBOX (w/h <= 0), "
                                     "yok sayıldı\n";
                        continue;
                    }

                    bbox_received.fetch_add(1, std::memory_order_relaxed);
                    std::cout << "[UartMavlinkLink] BBOX alındı: (" << bbox.x << ","
                              << bbox.y << "," << bbox.width << "," << bbox.height
                              << ")\n";
                    if (bbox_cb) bbox_cb(bbox);
                }
            }
        }
    }
};

UartMavlinkLink::UartMavlinkLink(std::string device, uint32_t baud)
    : impl_(std::make_unique<Impl>(std::move(device), baud)) {}

UartMavlinkLink::~UartMavlinkLink() { stop(); }

void UartMavlinkLink::setBboxCallback(BboxCallback cb) { impl_->bbox_cb = std::move(cb); }

bool UartMavlinkLink::start() {
    if (impl_->running.load()) return true;
    if (!impl_->openPort()) return false;

    impl_->running.store(true);
    impl_->read_thread = std::thread([this] { impl_->readLoop(); });
    std::cout << "[UartMavlinkLink] " << impl_->device << " dinleniyor (baud="
              << impl_->baud << ")\n";
    return true;
}

void UartMavlinkLink::stop() {
    if (!impl_->running.exchange(false)) return;
    if (impl_->read_thread.joinable()) impl_->read_thread.join();
    if (impl_->fd >= 0) {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

UartMavlinkLink::Stats UartMavlinkLink::getStats() const {
    Stats s;
    s.bytes_read = impl_->bytes_read.load(std::memory_order_relaxed);
    s.messages_parsed = impl_->messages_parsed.load(std::memory_order_relaxed);
    s.bbox_received = impl_->bbox_received.load(std::memory_order_relaxed);
    s.parse_errors = impl_->parse_errors.load(std::memory_order_relaxed);
    return s;
}