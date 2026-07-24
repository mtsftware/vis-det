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

    // Helper: align16
    auto align16 = [](uint32_t v) { return ((v + 15) / 16) * 16; };

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

        // ================================================================
        // ADIM 1: Decode cikti → RGA ile 640x640 RGB (inference icin)
        // ================================================================
        size_t rgb_model_size = static_cast<size_t>(model_w) * model_h * 3;
        DmaBufferPtr rgb_model = rga.acquireOutputBuffer(
            rgb_model_size, model_w, model_h, PixelFormat::RGB888);
        if (!rgb_model) {
            std::cerr << "[main] RGB model buffer tahsisi basarisiz.\n";
            return;
        }

        bool converted;
        if (detected_format == PixelFormat::NV16) {
            converted = rga.nv16ToRgb888(decoded_frame, original_width, original_height,
                                         rgb_model, model_w, model_h);
        } else {
            converted = rga.nv12ToRgb888(decoded_frame, original_width, original_height,
                                         rgb_model, model_w, model_h);
        }
        if (!converted) {
            std::cerr << "[main] YUV->RGB donusum basarisiz.\n";
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
        
        // Debug: output bilgisi
        std::cout << "[main] Output size: " << output_size 
                  << " bytes, raw_output: " << (raw_output ? "valid" : "NULL") << "\n";
        
        // Threshold geçici olarak 0.1'e düşürüldü (test amaçlı — model çok düşük skor veriyor olabilir)
        const float detection_conf_thresh = 0.1f;
        
        YoloPostProcessor::decodeOutputs(
            raw_output, output_size,
            original_width, original_height,
            model_w, model_h,
            detection_conf_thresh, detections);

        std::cout << "[main] decodeOutputs sonra: " << detections.size() << " detection.\n";

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
        // ADIM 5: Islenmis frame olustur (clone → RGB → draw → NV12)
        // ================================================================
        // Decoder cikti sınırlı havuzdan geliyor — yerinde çizim yapmayýÝ BIR.
        // Ayri bir kopya olustur.
        DmaBufferPtr working_copy;
        if (!rga.cloneFrame(decoded_frame,
                            (detected_format == PixelFormat::NV16) ? RgaPixelFormat::NV16
                                                                    : RgaPixelFormat::NV12,
                            working_copy)) {
            std::cerr << "[main] Frame clone basarisiz.\n";
            return;
        }

        // Working copy'i RGB'ye çevir (çizim için)
        uint32_t orig_aligned_w = align16(original_width);
        uint32_t orig_aligned_h = align16(original_height);
        size_t yuv_size;
        if (detected_format == PixelFormat::NV16) {
            yuv_size = static_cast<size_t>(orig_aligned_w) * orig_aligned_h * 2;
        } else {
            yuv_size = static_cast<size_t>(orig_aligned_w) * orig_aligned_h * 3 / 2;
        }

        DmaBufferPtr rgb_work = rga.acquireOutputBuffer(
            static_cast<uint32_t>(yuv_size), orig_aligned_w, orig_aligned_h,
            (detected_format == PixelFormat::NV16) ? PixelFormat::NV16 : PixelFormat::NV12);

        // Working NV12 → RGB (büyük boyutta)
        DmaBufferPtr rgb_large = rga.acquireOutputBuffer(
            static_cast<uint32_t>(orig_aligned_w * orig_aligned_h * 3),
            orig_aligned_w, orig_aligned_h, PixelFormat::RGB888);

        bool ok;
        if (detected_format == PixelFormat::NV16) {
            ok = rga.nv16ToRgb888(working_copy, orig_aligned_w, orig_aligned_h,
                                  rgb_large, orig_aligned_w, orig_aligned_h);
        } else {
            ok = rga.nv12ToRgb888(working_copy, orig_aligned_w, orig_aligned_h,
                                  rgb_large, orig_aligned_w, orig_aligned_h);
        }

        if (ok) {
            // Tespit edilen object'leri çiz
            if (!tracked_objects.empty()) {
                rga.drawDetectedObjects(rgb_large, orig_aligned_w, orig_aligned_h,
                                       tracked_objects, 2, true);
            }

            // RGB → NV12 geri dönüştür (encoder için)
            DmaBufferPtr yuv_processed = rga.acquireOutputBuffer(
                static_cast<uint32_t>(yuv_size), orig_aligned_w, orig_aligned_h,
                detected_format);

            if (yuv_processed) {
                bool back_converted;
                if (detected_format == PixelFormat::NV16) {
                    back_converted = rga.rgb888ToNv16(rgb_large, orig_aligned_w, orig_aligned_h,
                                                       yuv_processed, orig_aligned_w, orig_aligned_h);
                } else {
                    back_converted = rga.rgb888ToNv12(rgb_large, orig_aligned_w, orig_aligned_h,
                                                       yuv_processed, orig_aligned_w, orig_aligned_h);
                }

                if (back_converted) {
                    streamer.pushFrame(yuv_processed);
                }
            }
        }

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