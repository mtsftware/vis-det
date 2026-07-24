#include "pipeline/PipelineOrchestrator.hpp"

#include "decode/MppDecoder.hpp"
#include "inference/YoloInferenceEngine.hpp"
#include "ingestion/RtspDemuxer.hpp"
#include "postprocess/YoloPostProcessor.hpp"
#include "rga_processing/RgaPreprocessor.hpp"

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
    PipelineConfig config;

    ResultCallback result_cb;

    YoloInferenceEngine yolo;
    RgaPreprocessor rga;
    MultiObjectTracker tracker;

    std::unique_ptr<RtspDemuxer> demuxer;
    std::unique_ptr<MppDecoder> decoder;

    // En-yeni-kare slotu (tracking-app PipelineOrchestrator ile ayni desen):
    // decode callback'i buraya birakir ve DONER — asla bloklanmaz. Bayat
    // kare (islenmeden once yenisi gelirse) dusurulur.
    std::mutex frame_mtx;
    std::condition_variable frame_cv;
    DmaBufferPtr latest_frame;

    std::thread process_thread;
    std::atomic<bool> running{false};

    std::atomic<bool> first_frame_seen{false};
    std::atomic<uint32_t> source_width{1280};
    std::atomic<uint32_t> source_height{720};
    PixelFormat source_format = PixelFormat::Unknown;  // sadece isleme thread'i yazar

    std::atomic<uint64_t> frames_processed{0};
    std::atomic<uint64_t> frames_dropped{0};
    uint64_t frame_index = 0;  // sadece isleme thread'i dokunur

    explicit Impl(PipelineConfig cfg) : config(std::move(cfg)) {}

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

    void processingLoop() {
        PipelineOrchestrator::pinCurrentThreadToBigCores();

        RgaPixelFormat src_rga_fmt = RgaPixelFormat::NV12;

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
            if (!frame || !frame->virt_addr) continue;

            if (!first_frame_seen.load()) {
                source_width.store(frame->width);
                source_height.store(frame->height);
                source_format = frame->format;
                first_frame_seen.store(true);
                std::cout << "[PipelineOrchestrator] Ilk kare: "
                          << frame->width << "x" << frame->height
                          << " format=" << (frame->format == PixelFormat::NV16 ? "NV16" : "NV12")
                          << "\n";
            }
            src_rga_fmt = (frame->format == PixelFormat::NV16) ? RgaPixelFormat::NV16
                                                                : RgaPixelFormat::NV12;

            // ADIM 1: letterbox -> 640x640 RGB888 (NPU girdisi)
            DmaBufferPtr rgb_model;
            LetterboxResult letterbox;
            if (!rga.process(frame, src_rga_fmt, rgb_model, letterbox)) {
                std::cerr << "[PipelineOrchestrator] letterbox basarisiz, kare atlandi\n";
                continue;
            }

            // ADIM 2: YOLO inference (NPU)
            uint32_t input_size = static_cast<uint32_t>(rgb_model->size);
            if (!yolo.run(rgb_model->virt_addr, input_size)) {
                std::cerr << "[PipelineOrchestrator] YOLO inference basarisiz\n";
                continue;
            }

            // ADIM 3: output decode + DIoU-NMS
            const void* raw_output = nullptr;
            uint32_t output_size = 0;
            if (!yolo.getOutput(raw_output, output_size)) {
                std::cerr << "[PipelineOrchestrator] YOLO output alinamadi\n";
                yolo.releaseOutputs();
                continue;
            }

            std::vector<YoloDetection> detections;
            YoloPostProcessor::decodeOutputs(
                raw_output, output_size,
                static_cast<int>(source_width.load()), static_cast<int>(source_height.load()),
                letterbox, config.detection_conf_thresh, detections);

            if (!detections.empty()) {
                detections = YoloPostProcessor::applyNMS(detections, config.nms_iou_thresh);
            }
            yolo.releaseOutputs();

            // ADIM 4: multi-object tracking
            auto tracked_objects = tracker.update(detections, static_cast<int>(frame_index));

            // ADIM 5: clone + dogrudan YUV uzerine cizim (RGB round-trip yok)
            DmaBufferPtr draw_frame;
            if (!rga.cloneFrame(frame, src_rga_fmt, draw_frame)) {
                std::cerr << "[PipelineOrchestrator] frame clone basarisiz\n";
                continue;
            }
            if (!tracked_objects.empty()) {
                YoloPostProcessor::drawTrackedObjects(draw_frame, tracked_objects, 4);
            }

            frames_processed.fetch_add(1, std::memory_order_relaxed);

            FrameResult out;
            out.frame = draw_frame;
            out.tracked_objects = std::move(tracked_objects);
            out.detection_count = static_cast<int>(detections.size());
            out.frame_index = frame_index;

            if (result_cb) result_cb(out);

            ++frame_index;
        }
    }
};

PipelineOrchestrator::PipelineOrchestrator(PipelineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

PipelineOrchestrator::~PipelineOrchestrator() { stop(); }

void PipelineOrchestrator::setResultCallback(ResultCallback cb) {
    impl_->result_cb = std::move(cb);
}

bool PipelineOrchestrator::start() {
    if (impl_->running.load()) return true;

    if (!impl_->yolo.loadModel(impl_->config.model_path)) {
        std::cerr << "[PipelineOrchestrator] YOLO modeli yuklenemedi: "
                  << impl_->config.model_path << "\n";
        return false;
    }
    if (!impl_->rga.configure(impl_->config.model_width, impl_->config.model_height,
                               RgaPixelFormat::RGB888)) {
        std::cerr << "[PipelineOrchestrator] RgaPreprocessor configure basarisiz\n";
        return false;
    }

    MultiObjectTracker::Config tcfg;
    tcfg.iou_thresh = 0.4f;
    tcfg.max_lost_frames = 10;
    impl_->tracker.setConfig(tcfg);

    impl_->demuxer.reset(new RtspDemuxer(impl_->config.rtsp_url, impl_->config.rtsp_latency_ms));
    impl_->decoder.reset(new MppDecoder(impl_->config.coding));

    // KRITIK: decode callback'i sadece en-yeni-kare slotuna birakir ve
    // DONER — agir isleme (letterbox/NPU/postprocess/cizim) processingLoop
    // icinde AYRI thread'de. Bu, onceki tek-thread senkron akisin (decode
    // thread'ini butun pipeline suresince bloklayip MPP'nin 16 slotluk
    // decode havuzunu doldurarak takilmaya yol acan hatanin) duzeltmesidir.
    impl_->decoder->setFrameCallback(
        [this](DmaBufferPtr frame) { impl_->onFrame(std::move(frame)); });
    impl_->demuxer->setPacketCallback([this](EncodedPacket&& pkt) {
        impl_->decoder->feedPacket(std::move(pkt));
    });

    impl_->running.store(true);
    impl_->process_thread = std::thread([this] { impl_->processingLoop(); });

    if (!impl_->decoder->start()) {
        std::cerr << "[PipelineOrchestrator] MPP decoder baslatilamadi\n";
        stop();
        return false;
    }
    if (!impl_->demuxer->start()) {
        std::cerr << "[PipelineOrchestrator] demuxer baslatilamadi\n";
        stop();
        return false;
    }

    std::cout << "[PipelineOrchestrator] basladi: " << impl_->config.rtsp_url << "\n";
    return true;
}

void PipelineOrchestrator::stop() {
    impl_->running.store(false);
    impl_->frame_cv.notify_all();
    if (impl_->process_thread.joinable()) impl_->process_thread.join();

    // Sira onemli: once paket kaynagi (demuxer), sonra decoder.
    if (impl_->demuxer) impl_->demuxer->stop();
    if (impl_->decoder) impl_->decoder->stop();

    {
        std::lock_guard<std::mutex> lk(impl_->frame_mtx);
        impl_->latest_frame.reset();
    }

    impl_->demuxer.reset();
    impl_->decoder.reset();
    impl_->yolo.unload();
}

PipelineOrchestrator::Stats PipelineOrchestrator::getStats() const {
    Stats s;
    if (impl_->decoder) {
        s.frames_decoded = impl_->decoder->getStats().frames_decoded;
    }
    s.frames_processed = impl_->frames_processed.load(std::memory_order_relaxed);
    s.frames_dropped = impl_->frames_dropped.load(std::memory_order_relaxed);
    s.active_tracks = static_cast<uint32_t>(impl_->tracker.activeCount());
    s.total_tracked = static_cast<uint32_t>(impl_->tracker.totalTracked());
    s.lost_tracks = static_cast<uint32_t>(impl_->tracker.lostCount());
    return s;
}

bool PipelineOrchestrator::firstFrameSeen() const {
    return impl_->first_frame_seen.load();
}

uint32_t PipelineOrchestrator::sourceWidth() const {
    return impl_->source_width.load();
}

uint32_t PipelineOrchestrator::sourceHeight() const {
    return impl_->source_height.load();
}

PixelFormat PipelineOrchestrator::sourceFormat() const {
    return impl_->source_format;
}

void PipelineOrchestrator::pinCurrentThreadToBigCores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 4; c <= 7; ++c) CPU_SET(c, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        std::cerr << "[PipelineOrchestrator] UYARI: buyuk cekirdek affinity "
                     "ayarlanamadi (devam ediliyor)\n";
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
