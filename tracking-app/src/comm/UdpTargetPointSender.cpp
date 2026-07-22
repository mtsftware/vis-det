#include "comm/UdpTargetPointSender.hpp"

#include "mavlink/myformat/myformat.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>

struct UdpTargetPointSender::Impl {
    std::string host;
    uint16_t port;
    int fd = -1;
    sockaddr_in dest{};

    Impl(std::string h, uint16_t p) : host(std::move(h)), port(p) {}
};

UdpTargetPointSender::UdpTargetPointSender(std::string host, uint16_t port)
    : impl_(std::make_unique<Impl>(std::move(host), port)) {}

UdpTargetPointSender::~UdpTargetPointSender() { stop(); }

bool UdpTargetPointSender::start() {
    if (impl_->fd >= 0) return true;

    impl_->fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (impl_->fd < 0) {
        std::cerr << "[UdpTargetPointSender] socket() başarısız: "
                  << std::strerror(errno) << "\n";
        return false;
    }

    std::memset(&impl_->dest, 0, sizeof(impl_->dest));
    impl_->dest.sin_family = AF_INET;
    impl_->dest.sin_port = htons(impl_->port);

    if (inet_pton(AF_INET, impl_->host.c_str(), &impl_->dest.sin_addr) != 1) {
        // Sayısal IP değilse hostname olarak çözülmeyi dener.
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;
        addrinfo* res = nullptr;
        int rc = getaddrinfo(impl_->host.c_str(), nullptr, &hints, &res);
        if (rc != 0 || !res) {
            std::cerr << "[UdpTargetPointSender] host çözümlenemedi: " << impl_->host
                      << "\n";
            ::close(impl_->fd);
            impl_->fd = -1;
            return false;
        }
        impl_->dest.sin_addr =
            reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
        freeaddrinfo(res);
    }

    std::cout << "[UdpTargetPointSender] hedef: " << impl_->host << ":" << impl_->port
              << "\n";
    return true;
}

void UdpTargetPointSender::stop() {
    if (impl_->fd >= 0) {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

void UdpTargetPointSender::send(float x, float y) {
    if (impl_->fd < 0) return;

    mavlink::myformat::msg::TARGET_POINT packet{};
    packet.x = x;
    packet.y = y;

    mavlink::mavlink_message_t msg{};
    mavlink::MsgMap map(&msg);
    packet.serialize(map);
    mavlink::mavlink_finalize_message(&msg, /*system_id=*/1, /*component_id=*/1,
                                       packet.MIN_LENGTH, packet.LENGTH,
                                       packet.CRC_EXTRA);

    uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    uint16_t len = mavlink::mavlink_msg_to_send_buffer(buf, &msg);

    ssize_t sent = ::sendto(impl_->fd, buf, len, 0,
                             reinterpret_cast<sockaddr*>(&impl_->dest),
                             sizeof(impl_->dest));
    if (sent < 0) {
        std::cerr << "[UdpTargetPointSender] sendto başarısız: " << std::strerror(errno)
                  << "\n";
    }
}