#include "pipeline/PipelineOrchestrator.hpp"

#include "decode/MppDecoder.hpp"
#include "ingestion/RtspDemuxer.hpp"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

struct PipelineOrchestrator::Impl {
    ITracker& tracker;
    PipelineConfig config;

    ResultCallback result_cb;
    LostCallback lost_cb;

    std::unique_ptr<RtspDemuxer> demuxer;
    std::unique_ptr<MppDecoder> decoder;

    // En-yeni-kare slotu (CLAUDE.md kural #5): decode callback'i buraya
    // bırakır ve DÖNER — asla bloklanmaz. Bayat kare düşürülür.
    std::mutex frame_mtx;
    std::condition_variable frame_cv;
    DmaBufferPtr latest_frame;

    // requestInitialize kuyruğu (tek elemanlı): initialize() SADECE
    // tracking thread'inde koşar (ITracker tek-thread kuralı).
    std::mutex init_mtx;
    BBox pending_init_bbox;
    bool has_pending_init = false;

    std::thread track_thread;
    std::atomic<bool> running{false};
    bool tracker_initialized = false;  // sadece tracking thread'i dokunur

    std::atomic<uint64_t> frames_tracked{0};
    std::atomic<uint64_t> frames_dropped{0};
    std::atomic<uint64_t> frames_lost{0};
    std::atomic<float> last_confidence{0.0f};
    uint32_t consecutive_lost = 0;  // sadece tracking thread'i dokunur

    explicit Impl(ITracker& t, PipelineConfig cfg)
        : tracker(t), config(std::move(cfg)) {}

    void onFrame(DmaBufferPtr frame) {
        {
            std::lock_guard<std::mutex> lk(frame_mtx);
            if (latest_frame) {
                frames_dropped.fetch_add(1, std::memory_order_relaxed);
            }
            latest_frame = std::move(frame);
        }
        frame_cv.notify_one();
    }

    void trackingLoop() {
        PipelineOrchestrator::pinCurrentThreadToBigCores();

        while (running.load()) {
            DmaBufferPtr frame;
            {
                std::unique_lock<std::mutex> lk(frame_mtx);
                frame_cv.wait_for(lk, std::chrono::milliseconds(100), [&] {
                    return latest_frame != nullptr || !running.load();
                });
                if (!latest_frame) continue;
                frame = std::move(latest_frame);
            }

            // Bekleyen initialize isteği bu karede işlenir (ilk hedef ya da
            // detector yeniden-edinimi). Başarısızsa istek DÜŞER — çağıran
            // LOST sinyali almaya devam edeceği için yeniden deneyebilir.
            bool do_init = false;
            BBox init_bbox;
            {
                std::lock_guard<std::mutex> lk(init_mtx);
                if (has_pending_init) {
                    do_init = true;
                    init_bbox = pending_init_bbox;
                    has_pending_init = false;
                }
            }
            if (do_init) {
                if (tracker.initialize(frame, init_bbox)) {
                    tracker_initialized = true;
                    consecutive_lost = 0;
                    std::cout << "[PipelineOrchestrator] tracker initialize edildi, "
                                 "bbox=(" << init_bbox.x << "," << init_bbox.y << ","
                              << init_bbox.width << "," << init_bbox.height << ")\n";
                } else {
                    std::cerr << "[PipelineOrchestrator] tracker.initialize() "
                                 "başarısız — istek düşürüldü\n";
                }
            }

            FrameResult out;
            out.frame = frame;

            if (tracker_initialized) {
                out.track = tracker.track(frame);
                out.tracked = true;
                out.state = tracker.getState();

                frames_tracked.fetch_add(1, std::memory_order_relaxed);
                last_confidence.store(out.track.confidence, std::memory_order_relaxed);

                if (out.state == TrackerState::Lost) {
                    frames_lost.fetch_add(1, std::memory_order_relaxed);
                    ++consecutive_lost;
                    if (lost_cb) lost_cb(consecutive_lost);
                } else {
                    consecutive_lost = 0;
                }
            } else {
                // Henüz hedef yok: ham kare yine de dışarı verilir (yayın
                // kesintisiz aksın, kullanıcı görüntü üzerinden bbox seçebilsin).
                out.tracked = false;
                out.state = TrackerState::Idle;
            }

            if (result_cb) result_cb(out);
        }
    }
};

PipelineOrchestrator::PipelineOrchestrator(ITracker& tracker, PipelineConfig config)
    : impl_(std::make_unique<Impl>(tracker, std::move(config))) {}

PipelineOrchestrator::~PipelineOrchestrator() { stop(); }

void PipelineOrchestrator::setResultCallback(ResultCallback cb) {
    impl_->result_cb = std::move(cb);
}

void PipelineOrchestrator::setLostCallback(LostCallback cb) {
    impl_->lost_cb = std::move(cb);
}

bool PipelineOrchestrator::start() {
    if (impl_->running.load()) return true;

    impl_->demuxer.reset(new RtspDemuxer(impl_->config.rtsp_url,
                                          impl_->config.rtsp_latency_ms));
    impl_->decoder.reset(new MppDecoder(impl_->config.coding));

    impl_->decoder->setFrameCallback(
        [this](DmaBufferPtr frame) { impl_->onFrame(std::move(frame)); });
    impl_->demuxer->setPacketCallback([this](EncodedPacket&& pkt) {
        impl_->decoder->feedPacket(std::move(pkt));
    });

    impl_->running.store(true);
    impl_->track_thread = std::thread([this] { impl_->trackingLoop(); });

    if (!impl_->decoder->start()) {
        std::cerr << "[PipelineOrchestrator] MPP decoder başlatılamadı\n";
        stop();
        return false;
    }
    if (!impl_->demuxer->start()) {
        std::cerr << "[PipelineOrchestrator] demuxer başlatılamadı\n";
        stop();
        return false;
    }

    std::cout << "[PipelineOrchestrator] başladı: " << impl_->config.rtsp_url << "\n";
    return true;
}

void PipelineOrchestrator::stop() {
    if (!impl_->running.exchange(false)) {
        // Tracking thread hiç başlamamış olabilir (start hatası) — yine de
        // demuxer/decoder temizliği aşağıda koşmalı.
    }
    impl_->frame_cv.notify_all();
    if (impl_->track_thread.joinable()) impl_->track_thread.join();

    // Sıra önemli: önce paket kaynağı (demuxer), sonra decoder — decoder
    // durdurulduktan sonra paket beslemek MPP tarafında hataya düşer.
    if (impl_->demuxer) impl_->demuxer->stop();
    if (impl_->decoder) impl_->decoder->stop();

    // Slottaki son kare bırakılır (MPP buffer'ı decoder'a geri döner).
    {
        std::lock_guard<std::mutex> lk(impl_->frame_mtx);
        impl_->latest_frame.reset();
    }

    impl_->demuxer.reset();
    impl_->decoder.reset();
}

void PipelineOrchestrator::requestInitialize(const BBox& bbox) {
    std::lock_guard<std::mutex> lk(impl_->init_mtx);
    impl_->pending_init_bbox = bbox;
    impl_->has_pending_init = true;
}

PipelineOrchestrator::Stats PipelineOrchestrator::getStats() const {
    Stats s;
    if (impl_->decoder) {
        s.frames_decoded = impl_->decoder->getStats().frames_decoded;
    }
    s.frames_tracked = impl_->frames_tracked.load(std::memory_order_relaxed);
    s.frames_dropped = impl_->frames_dropped.load(std::memory_order_relaxed);
    s.frames_lost = impl_->frames_lost.load(std::memory_order_relaxed);
    s.last_confidence = impl_->last_confidence.load(std::memory_order_relaxed);
    return s;
}

void PipelineOrchestrator::pinCurrentThreadToBigCores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 4; c <= 7; ++c) CPU_SET(c, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        std::cerr << "[PipelineOrchestrator] UYARI: büyük çekirdek affinity "
                     "ayarlanamadı (devam ediliyor)\n";
    }
}

void PipelineOrchestrator::restoreCurrentThreadAllCores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    long n = sysconf(_SC_NPROCESSORS_CONF);
    if (n < 1) n = 8;
    for (long c = 0; c < n; ++c) CPU_SET(static_cast<int>(c), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}