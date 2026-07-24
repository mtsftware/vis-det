#include "decode/MppDecoder.hpp"

// Rockchip MPP native API headers (sadece RK3588 hedefinde derlenir)
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "rk_mpi.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace {
constexpr int kMaxFramePoolCount = 16;
constexpr int kMaxPutRetries = 50;

MppCodingType toMppCoding(VideoCoding c) {
    switch (c) {
        case VideoCoding::H264:
            return MPP_VIDEO_CodingAVC;
        case VideoCoding::H265:
            return MPP_VIDEO_CodingHEVC;
    }
    return MPP_VIDEO_CodingAVC;
}
}  // namespace

struct MppDecoder::Impl {
    MppCodingType coding;

    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    MppBufferGroup frm_grp = nullptr;

    std::atomic<bool> running{false};
    std::thread put_thread;
    std::thread get_thread;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::queue<EncodedPacket> packet_queue;

    FrameCallback callback;

    std::atomic<uint64_t> frames_decoded{0};
    std::atomic<uint64_t> frames_dropped{0};
    std::atomic<uint64_t> packets_failed{0};
    std::atomic<uint64_t> last_decode_us{0};

    ~Impl() { stop(); }

    bool start() {
        MPP_RET ret = mpp_create(&ctx, &mpi);
        if (ret != MPP_OK) {
            std::cerr << "[MppDecoder] mpp_create başarısız: " << ret << "\n";
            return false;
        }

        ret = mpp_init(ctx, MPP_CTX_DEC, coding);
        if (ret != MPP_OK) {
            std::cerr << "[MppDecoder] mpp_init başarısız: " << ret << "\n";
            return false;
        }

        running.store(true);
        put_thread = std::thread([this] { putPacketLoop(); });
        get_thread = std::thread([this] { getFrameLoop(); });
        return true;
    }

    void stop() {
        if (!running.exchange(false)) return;

        queue_cv.notify_all();
        if (put_thread.joinable()) put_thread.join();
        if (get_thread.joinable()) get_thread.join();

        if (ctx) {
            mpp_destroy(ctx);
            ctx = nullptr;
            mpi = nullptr;
        }
        if (frm_grp) {
            mpp_buffer_group_put(frm_grp);
            frm_grp = nullptr;
        }
    }

    void putPacketLoop() {
        while (running.load()) {
            EncodedPacket packet;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock,
                              [this] { return !packet_queue.empty() || !running.load(); });
                if (!running.load() && packet_queue.empty()) break;
                if (packet_queue.empty()) continue;
                packet = std::move(packet_queue.front());
                packet_queue.pop();
            }

            MppPacket mpp_pkt = nullptr;
            mpp_packet_init(&mpp_pkt, packet.data.data(), packet.data.size());
            mpp_packet_set_pts(mpp_pkt, static_cast<RK_S64>(packet.pts_ns));

            MPP_RET ret;
            int retry = 0;
            do {
                ret = mpi->decode_put_packet(ctx, mpp_pkt);
                if (ret == MPP_ERR_BUFFER_FULL) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    ++retry;
                }
            } while (ret == MPP_ERR_BUFFER_FULL && retry < kMaxPutRetries);

            mpp_packet_deinit(&mpp_pkt);

            if (ret != MPP_OK) {
                packets_failed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void getFrameLoop() {
        while (running.load()) {
            auto t0 = std::chrono::steady_clock::now();
            MppFrame frame = nullptr;
            MPP_RET ret = mpi->decode_get_frame(ctx, &frame);

            if (ret != MPP_OK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            if (!frame) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            if (mpp_frame_get_info_change(frame)) {
                handleInfoChange(frame);
                mpp_frame_deinit(&frame);
                continue;
            }

            if (mpp_frame_get_eos(frame)) {
                mpp_frame_deinit(&frame);
                break;
            }

            if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
                mpp_frame_deinit(&frame);
                frames_dropped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // decode_get_frame()'in bu BASARILI cagrisinin surdugu sure
            // (bkz. Stats::last_decode_ms notu — ASYNC put/get mimarisi
            // yuzunden "saf decode suresi" degil ama darbogaz teshisi icin
            // kullanisli bir proxy).
            auto t1 = std::chrono::steady_clock::now();
            uint64_t us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            last_decode_us.store(us, std::memory_order_relaxed);

            emitFrame(frame);
        }
    }

    void handleInfoChange(MppFrame frame) {
        uint32_t width = mpp_frame_get_width(frame);
        uint32_t height = mpp_frame_get_height(frame);
        uint32_t hor_stride = mpp_frame_get_hor_stride(frame);
        uint32_t ver_stride = mpp_frame_get_ver_stride(frame);

        std::cout << "[MppDecoder] Info change: " << width << "x" << height
                  << " (stride " << hor_stride << "x" << ver_stride << ")\n";

        if (!frm_grp) {
            MPP_RET grp_ret = mpp_buffer_group_get_internal(
                &frm_grp, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_DMA32);
            if (grp_ret != MPP_OK) {
                std::cerr << "[MppDecoder] buffer group oluşturulamadı: " << grp_ret
                          << "\n";
                return;
            }
            mpp_buffer_group_limit_config(frm_grp, 0, kMaxFramePoolCount);
        }

        mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp);
        mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
    }

    RK_U32 last_logged_fmt = 0xFFFFFFFF;

    void emitFrame(MppFrame frame) {
        MppFrameFormat raw_fmt = mpp_frame_get_fmt(frame);
        RK_U32 base_fmt = raw_fmt & MPP_FRAME_FMT_MASK;
        if (base_fmt != last_logged_fmt) {
            std::cout << "[MppDecoder] frame fmt: raw=0x" << std::hex << raw_fmt
                      << std::dec << " base=" << base_fmt << "\n";
            last_logged_fmt = base_fmt;
        }

        MppBuffer mpp_buf = mpp_frame_get_buffer(frame);
        if (!mpp_buf) {
            mpp_frame_deinit(&frame);
            return;
        }

        auto buf = std::make_shared<DmaBuffer>();
        buf->fd = mpp_buffer_get_fd(mpp_buf);
        buf->virt_addr = mpp_buffer_get_ptr(mpp_buf);
        buf->size = static_cast<uint32_t>(mpp_buffer_get_size(mpp_buf));
        buf->width = mpp_frame_get_width(frame);
        buf->height = mpp_frame_get_height(frame);
        buf->w_stride = mpp_frame_get_hor_stride(frame);
        buf->h_stride = mpp_frame_get_ver_stride(frame);
        buf->format = (base_fmt == MPP_FMT_YUV422SP) ? PixelFormat::NV16
                                                       : PixelFormat::NV12;
        buf->dtype = DataType::UINT8;
        buf->pts_ns = static_cast<uint64_t>(mpp_frame_get_pts(frame));

        buf->release_fn = [frame]() mutable { mpp_frame_deinit(&frame); };

        frames_decoded.fetch_add(1, std::memory_order_relaxed);

        if (callback) {
            callback(buf);
        }
    }
};

MppDecoder::MppDecoder(VideoCoding coding) : impl_(std::make_unique<Impl>()) {
    impl_->coding = toMppCoding(coding);
}

MppDecoder::~MppDecoder() { stop(); }

void MppDecoder::setFrameCallback(FrameCallback cb) { impl_->callback = std::move(cb); }

bool MppDecoder::start() { return impl_->start(); }

void MppDecoder::stop() {
    if (impl_) impl_->stop();
}

void MppDecoder::feedPacket(EncodedPacket&& packet) {
    {
        std::lock_guard<std::mutex> lock(impl_->queue_mutex);
        impl_->packet_queue.push(std::move(packet));
    }
    impl_->queue_cv.notify_one();
}

MppDecoder::Stats MppDecoder::getStats() const {
    return Stats{
        impl_->frames_decoded.load(std::memory_order_relaxed),
        impl_->frames_dropped.load(std::memory_order_relaxed),
        impl_->packets_failed.load(std::memory_order_relaxed),
        static_cast<double>(impl_->last_decode_us.load(std::memory_order_relaxed)) / 1000.0,
    };
}