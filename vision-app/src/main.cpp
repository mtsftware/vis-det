#include "ingestion/RtspDemuxer.hpp"
#include "decode/MppDecoder.hpp"
#include "streaming/RtspStreamer.hpp"
#include "rga_processing/RgaPreprocessor.hpp"
#include "inference/YoloInferenceEngine.hpp"
#include "postprocess/YoloPostProcessor.hpp"
#include "tracking/MultiObjectTracker.hpp"
#include "types/DmaBuffer.hpp"
#include "types/VideoCoding.hpp"

#include <csignal>
#include <iostream>
#include <mutex>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <memory>

static std::atomic<bool> g_running{true};
static std::mutex g_frame_mutex;

void signalHandler(int /*signum*/) {
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::string rtsp_url = "rtsp://localhost:8554/video";
    if (argc > 1) {
        rtsp_url = argv[1];
    }

    std::cout << "============================================================\n";
    std::cout << "  Vision App - YOLOv8n + Multi-Object Tracking Pipeline\n";
    std::cout << "============================================================\n";
    std::cout << "RTSP Kaynak : " << rtsp_url << "\n";
    std::cout << "Yayin URL   : rtsp://<board-ip>:8554/live\n";
    std::cout << "============================================================\n";

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    const uint32_t model_w = 640;
    const uint32_t model_h = 640;
    const uint32_t fps = 30;

    // ==================== YOLO INFERENCE KURULUMU ====================
    std::string model_path = "models/best-rk3588.rknn";

    YoloInferenceEngine yolo_engine;
    if (!yolo_engine.loadModel(model_path)) {
        std::cerr << "[main] YOLO modeli yuklenemedi: " << model_path << "\n";
        return 1;
    }
    std::cout << "[main] YOLOv8n NPU modeli yuklendi.\n";

    // Tracker kurulumu
    MultiObjectTracker tracker;
    MultiObjectTracker::Config tcfg;
    tcfg.iou_thresh = 0.4f;
    tcfg.max_lost_frames = 10;
    tracker.setConfig(tcfg);
    std::cout << "[main] Multi-Object Tracker kuruldu.\n";

    // ==================== RTSP PIPELINE KURULUMU ====================

    RtspDemuxer demuxer(rtsp_url, 5000);
    if (!demuxer.start()) {
        std::cerr << "[main] Demuxer baslatilamadi!\n";
        return 1;
    }
    std::cout << "[main] Demuxer basarili.\n";

    MppDecoder decoder(VideoCoding::H264);
    if (!decoder.start()) {
        std::cerr << "[main] Decoder baslatilamadi!\n";
        return 1;
    }
    std::cout << "[main] Decoder basarili.\n";

    uint32_t original_width = 1280;
    uint32_t original_height = 720;
    PixelFormat detected_format = PixelFormat::Unknown;
    std::atomic<int> frame_count{0};

    RtspStreamer streamer(8554, "/live", original_width, original_height, fps);
    if (!streamer.start()) {
        std::cerr << "[main] Streamer baslatilamadi!\n";
        return 1;
    }
    std::cout << "[main] Streamer basarili (NV12/NV16).\n";

    RgaPreprocessor rga;
    // Referans: tracking-app RgaPreprocessor::configure() ile ayni kalip —
    // hedef 640x640 RGB888 (YOLO NHWC girdisi), bir kez configure edilir.
    if (!rga.configure(model_w, model_h, RgaPixelFormat::RGB888)) {
        std::cerr << "[main] RgaPreprocessor configure basarisiz.\n";
        return 1;
    }

    // ==================== FRAME ISLEME CALLBACK ====================
    decoder.setFrameCallback([&](DmaBufferPtr decoded_frame) {
        if (!g_running.load()) return;
        if (!decoded_frame || !decoded_frame->virt_addr) return;

        int cnt = frame_count.fetch_add(1);

        // İlk frame'de format ve boyut tespit et
        if (cnt == 0) {
            original_width = decoded_frame->width;
            original_height = decoded_frame->height;
            detected_format = decoded_frame->format;
            streamer.setDimensions(original_width, original_height);

            std::cout << "[main] Orijinal boyutlar: "
                      << original_width << "x" << original_height << "\n";
            std::cout << "[main] Format: "
                      << (detected_format == PixelFormat::NV16 ? "NV16" : "NV12") << "\n";
        }

        std::lock_guard<std::mutex> lock(g_frame_mutex);

        RgaPixelFormat src_rga_fmt = (detected_format == PixelFormat::NV16)
                                          ? RgaPixelFormat::NV16
                                          : RgaPixelFormat::NV12;

        // ================================================================
        // ADIM 1: Decode cikti → RGA letterbox ile 640x640 RGB (inference icin)
        // Referans: RgaPreprocessor::process() (tracking-app ile ayni kalip),
        // aspect-ratio korunur, geri-map icin LetterboxResult donduru.
        // ================================================================
        DmaBufferPtr rgb_model;
        LetterboxResult letterbox;
        if (!rga.process(decoded_frame, src_rga_fmt, rgb_model, letterbox)) {
            std::cerr << "[main] Letterbox preprocess basarisiz.\n";
            return;
        }

        // ================================================================
        // ADIM 2: YOLO INFERENCE (NPU)
        // ================================================================
        uint32_t input_size = static_cast<uint32_t>(rgb_model->size);
        if (!yolo_engine.run(rgb_model->virt_addr, input_size)) {
            std::cerr << "[main] YOLO inference basarisiz.\n";
            return;
        }

        // ================================================================
        // ADIM 3: OUTPUT DECODE + NMS
        // ================================================================
        const void* raw_output = nullptr;
        uint32_t output_size = 0;
        if (!yolo_engine.getOutput(raw_output, output_size)) {
            std::cerr << "[main] YOLO output alinamadi.\n";
            yolo_engine.releaseOutputs();
            return;
        }

        std::vector<YoloDetection> detections;
        const float detection_conf_thresh = 0.4f;

        YoloPostProcessor::decodeOutputs(
            raw_output, output_size,
            original_width, original_height,
            letterbox,
            detection_conf_thresh, detections);

        if (!detections.empty()) {
            detections = YoloPostProcessor::applyNMS(detections, 0.45f);
        }
        yolo_engine.releaseOutputs();

        if (cnt % 30 == 0) {
            std::cout << "[main] Frame " << cnt << ": " << detections.size()
                      << " adet tespit.\n";
        }

        // ================================================================
        // ADIM 4: MULTI-OBJECT TRACKING
        // ================================================================
        auto tracked_objects = tracker.update(detections, cnt);

        if (cnt % 30 == 0) {
            std::cout << "[main]  Aktif track: " << tracked_objects.size()
                      << " (toplam: " << tracker.totalTracked()
                      << ", kaybolan: " << tracker.lostCount() << ")\n";
        }

        // ================================================================
        // ADIM 5: Islenmis frame olustur (clone → cizim dogrudan YUV uzerinde)
        // Referans: tracking-app drawBboxOutline() ile ayni teknik — RGB
        // donusumune GEREK YOK, RGA native NV12/NV16 uzerine dogrudan cizer.
        // ================================================================
        DmaBufferPtr draw_frame;
        if (!rga.cloneFrame(decoded_frame, src_rga_fmt, draw_frame)) {
            std::cerr << "[main] Frame clone basarisiz.\n";
            return;
        }

        if (!tracked_objects.empty()) {
            YoloPostProcessor::drawTrackedObjects(draw_frame, tracked_objects, 4);
        }

        streamer.pushFrame(draw_frame);

        if ((cnt+1) % 100 == 0) {
            std::cout << "[main] " << (cnt+1) << " islenmis frame stream'e gonderildi.\n";
        }
    });

    // Demuxer callback
    demuxer.setPacketCallback([&](EncodedPacket&& packet) {
        if (!g_running.load()) return;
        decoder.feedPacket(std::move(packet));
    });

    std::cout << "[main] Pipeline baslatildi. Cikmak icin Ctrl+C basin.\n";
    std::cout << "[main] Yayin: rtsp://<board-ip>:8554/live\n";

    // Main loop
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto stats = decoder.getStats();
        if (stats.frames_decoded % 300 == 0 && stats.frames_decoded > 0) {
            std::cout << "[main] Decoder: " << stats.frames_decoded
                      << " decoded, " << stats.frames_dropped << " dropped\n";
        }
    }

    std::cout << "[main] Pipeline durduruluyor...\n";

    decoder.stop();
    demuxer.stop();
    streamer.stop();
    yolo_engine.unload();

    auto final_stats = decoder.getStats();
    std::cout << "[main] Toplam " << final_stats.frames_decoded
              << " frame cozuldu.\n";
    std::cout << "[main] Pipeline kapatildi.\n";

    return 0;
}