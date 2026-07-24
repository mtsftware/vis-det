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
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

namespace {

// TESHIS YARDIMCISI: NPU'ya giden letterbox'lanmis RGB888 karesini binary
// PPM (P6) olarak diske yazar — "0 tespit" durumunda modele giden goruntunun
// gercekten dogru (renk sirasi RGB mi BGR mi, letterbox geometrisi dogru mu)
// olup olmadigini GOZLE dogrulamak icin. Sadece bir kez (ilk kare) cagrilir,
// uretimde performans etkisi yok. `scp board:/tmp/vision_app_debug_input.ppm .`
// ile indirip acabilirsiniz (GIMP/feh/vs ppm'i dogrudan acar).
void dumpRgbDebugPpm(const DmaBufferPtr& rgb, const char* path) {
    if (!rgb || !rgb->virt_addr) {
        std::cerr << "[PipelineOrchestrator] debug PPM: virt_addr yok, atlaniyor\n";
        return;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[PipelineOrchestrator] debug PPM acilamadi: " << path << "\n";
        return;
    }
    f << "P6\n" << rgb->width << " " << rgb->height << "\n255\n";
    f.write(static_cast<const char*>(rgb->virt_addr),
            static_cast<std::streamsize>(static_cast<size_t>(rgb->width) * rgb->height * 3));
    std::cout << "[PipelineOrchestrator] NPU girdisi diske yazildi: " << path
              << " (" << rgb->width << "x" << rgb->height << ", RGB888)\n";
}

}  // namespace

struct PipelineOrchestrator::Impl {
    PipelineConfig config;

    ResultCallback result_cb;

    YoloInferenceEngine yolo;
    RgaPreprocessor rga;
    ByteTracker tracker;

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
    std::atomic<uint64_t> last_inference_us{0};    // yolo.run()+fetchOutputs() wall-clock
    std::atomic<uint64_t> last_letterbox_us{0};    // rga.process()
    std::atomic<uint64_t> last_postprocess_us{0};  // decode+NMS+ByteTrack
    std::atomic<uint64_t> last_draw_us{0};         // cloneFrame()+drawTrackedObjects()
    uint64_t frame_index = 0;  // sadece isleme thread'i dokunur

    std::vector<std::string> labels;  // coco_labels.txt, start()'ta yuklenir

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
            auto lb_t0 = std::chrono::steady_clock::now();
            DmaBufferPtr rgb_model;
            LetterboxResult letterbox;
            if (!rga.process(frame, src_rga_fmt, rgb_model, letterbox)) {
                std::cerr << "[PipelineOrchestrator] letterbox basarisiz, kare atlandi\n";
                continue;
            }
            auto lb_t1 = std::chrono::steady_clock::now();
            last_letterbox_us.store(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(lb_t1 - lb_t0).count()),
                std::memory_order_relaxed);

            // TESHIS: NPU'ya giden gercek letterbox'lanmis goruntuyu sadece
            // ilk karede diske yaz (bkz. dosya basindaki dumpRgbDebugPpm notu)
            // — "0 tespit" teshisinde renk/geometri dogrulamasi icin.
            if (frame_index == 0) {
                dumpRgbDebugPpm(rgb_model, "/tmp/vision_app_debug_input.ppm");
            }

            // ADIM 2: YOLO inference (NPU) — run()+fetchOutputs() wall-clock
            // olcumu (main.cpp'nin periyodik ozet satiri icin, Stats::inference_ms).
            auto infer_t0 = std::chrono::steady_clock::now();

            uint32_t input_size = static_cast<uint32_t>(rgb_model->size);
            if (!yolo.run(rgb_model->virt_addr, input_size)) {
                std::cerr << "[PipelineOrchestrator] YOLO inference basarisiz\n";
                continue;
            }

            // ADIM 3: 9 cikti tensorunu TEK cagriyla al, DFL decode + DIoU-NMS
            if (!yolo.fetchOutputs()) {
                std::cerr << "[PipelineOrchestrator] YOLO ciktilari alinamadi\n";
                yolo.releaseOutputs();
                continue;
            }

            auto infer_t1 = std::chrono::steady_clock::now();
            last_inference_us.store(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(infer_t1 - infer_t0)
                        .count()),
                std::memory_order_relaxed);

            auto post_t0 = std::chrono::steady_clock::now();

            std::vector<YoloDetection> detections;
            float max_score = -1.0f;
            YoloPostProcessor::decodeOutputs(
                yolo,
                static_cast<int>(source_width.load()), static_cast<int>(source_height.load()),
                letterbox, config.detection_conf_thresh, detections, &max_score);

            if (frame_index % 30 == 0) {
                std::cout << "[PipelineOrchestrator] frame " << frame_index
                          << ": max_score(esiksiz)=" << max_score
                          << " esik=" << config.detection_conf_thresh
                          << " ham_tespit=" << detections.size() << "\n";
            }

            if (!detections.empty()) {
                detections = YoloPostProcessor::applyNMS(detections, config.nms_iou_thresh);
            }
            yolo.releaseOutputs();

            // ADIM 4: ByteTrack — kaybolan track SADECE bir sonraki update()
            // cagrisinda ciktidan duser (bkz. ByteTracker.hpp), ekstra bir
            // "lost_count/max_lost_frames" bekletme mantigina GEREK YOK.
            auto tracked_objects = tracker.update(detections);

            auto post_t1 = std::chrono::steady_clock::now();
            last_postprocess_us.store(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(post_t1 - post_t0)
                        .count()),
                std::memory_order_relaxed);

            // ADIM 5: clone + dogrudan YUV uzerine cizim (RGB round-trip yok)
            auto draw_t0 = std::chrono::steady_clock::now();
            DmaBufferPtr draw_frame;
            if (!rga.cloneFrame(frame, src_rga_fmt, draw_frame)) {
                std::cerr << "[PipelineOrchestrator] frame clone basarisiz\n";
                continue;
            }
            if (!tracked_objects.empty()) {
                YoloPostProcessor::drawTrackedObjects(draw_frame, tracked_objects, 4);
            }
            auto draw_t1 = std::chrono::steady_clock::now();
            last_draw_us.store(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(draw_t1 - draw_t0)
                        .count()),
                std::memory_order_relaxed);

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
    impl_->labels = YoloPostProcessor::loadLabels(impl_->config.labels_path);
    std::cout << "[PipelineOrchestrator] " << impl_->labels.size()
              << " sinif etiketi yuklendi: " << impl_->config.labels_path << "\n";

    if (!impl_->rga.configure(impl_->config.model_width, impl_->config.model_height,
                               RgaPixelFormat::RGB888)) {
        std::cerr << "[PipelineOrchestrator] RgaPreprocessor configure basarisiz\n";
        return false;
    }

    // ByteTracker'in kendi config'i tepe-duzey ByteTrackConfig namespace'inde
    // (bkz. include/tracking/ByteTracker.hpp) — burada ayrica setConfig
    // cagrisina gerek yok.

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
        auto dstats = impl_->decoder->getStats();
        s.frames_decoded = dstats.frames_decoded;
        s.decode_ms = dstats.last_decode_ms;
    }
    s.frames_processed = impl_->frames_processed.load(std::memory_order_relaxed);
    s.frames_dropped = impl_->frames_dropped.load(std::memory_order_relaxed);
    s.active_tracks = static_cast<uint32_t>(impl_->tracker.activeCount());
    s.total_tracked = static_cast<uint32_t>(impl_->tracker.totalTracked());
    s.lost_tracks = static_cast<uint32_t>(impl_->tracker.lostCount());
    s.inference_ms = static_cast<double>(impl_->last_inference_us.load(std::memory_order_relaxed)) / 1000.0;
    s.letterbox_ms = static_cast<double>(impl_->last_letterbox_us.load(std::memory_order_relaxed)) / 1000.0;
    s.postprocess_ms = static_cast<double>(impl_->last_postprocess_us.load(std::memory_order_relaxed)) / 1000.0;
    s.draw_ms = static_cast<double>(impl_->last_draw_us.load(std::memory_order_relaxed)) / 1000.0;
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

const std::vector<std::string>& PipelineOrchestrator::labels() const {
    return impl_->labels;
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
