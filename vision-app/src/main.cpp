#include "pipeline/PipelineOrchestrator.hpp"
#include "streaming/RtspStreamer.hpp"
#include "types/DmaBuffer.hpp"
#include "types/VideoCoding.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

// Referans: tracking-app/src/pipeline/main_m11_test.cpp ile ayni kalip —
// main() sadece orkestratör kurar + ResultCallback icinde HAFIF is
// (streamer.pushFrame) yapar. Agir pipeline (letterbox/NPU/postprocess/
// tracking/cizim) artik PipelineOrchestrator'in kendi isleme thread'inde,
// decode thread'inden AYRI — onceki surumdeki takilma/tikanma sebebiydi
// (bkz. PipelineOrchestrator.cpp basindaki not).

static std::atomic<bool> g_running{true};

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
    std::cout << "Yayin URL   : rtsp://<board-ip>:8557/out\n";
    std::cout << "============================================================\n";

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    const uint32_t fps = 30;

    // Cikis portu GIRDI kaynagindan (rtsp_url, orn. ffmpeg'in dinledigi
    // 8554) BILEREK farkli — ayni makinede ikisi de 8554'e bind etmeye
    // calisirsa cakisir (referans: tracking-app main_m11_test.cpp, giris
    // degisken/cikis hep 8557/out).
    RtspStreamer streamer(8557, "/out", 1280, 720, fps);
    if (!streamer.start()) {
        std::cerr << "[main] Streamer baslatilamadi!\n";
        return 1;
    }
    std::cout << "[main] Streamer basarili (NV12/NV16).\n";

    PipelineConfig config;
    config.rtsp_url = rtsp_url;
    config.rtsp_latency_ms = 5000;
    config.coding = VideoCoding::H264;
    config.model_path = "models/yolov8n_int8.rknn";
    config.labels_path = "models/coco_labels.txt";
    config.model_width = 640;
    config.model_height = 640;
    config.detection_conf_thresh = 0.4f;
    config.nms_iou_thresh = 0.45f;

    PipelineOrchestrator orchestrator(config);

    bool dims_set = false;
    orchestrator.setResultCallback([&](const FrameResult& result) {
        if (!g_running.load()) return;

        // Kaynak boyutu ilk karede netlesir — streamer'a bir kez bildir.
        if (!dims_set && orchestrator.firstFrameSeen()) {
            streamer.setDimensions(orchestrator.sourceWidth(), orchestrator.sourceHeight());
            dims_set = true;
            std::cout << "[main] Kaynak boyutu: " << orchestrator.sourceWidth() << "x"
                      << orchestrator.sourceHeight() << "\n";
        }

        // HAFIF is: sadece stream'e gonder (orkestratorun kendi thread'inde
        // calisir — burada agir islem YAPILMAZ, bkz. header notu).
        streamer.pushFrame(result.frame);

        if (result.frame_index % 30 == 0) {
            std::cout << "[main] Frame " << result.frame_index << ": "
                      << result.detection_count << " tespit, "
                      << result.tracked_objects.size() << " aktif track";
            const auto& labels = orchestrator.labels();
            for (const auto& obj : result.tracked_objects) {
                int cls = obj.class_id;
                std::cout << " [#" << obj.track_id << " "
                          << (cls >= 0 && static_cast<size_t>(cls) < labels.size()
                                  ? labels[static_cast<size_t>(cls)]
                                  : std::to_string(cls))
                          << " " << obj.confidence << "]";
            }
            std::cout << "\n";
        }
    });

    if (!orchestrator.start()) {
        std::cerr << "[main] Pipeline baslatilamadi!\n";
        streamer.stop();
        return 1;
    }

    std::cout << "[main] Pipeline baslatildi. Cikmak icin Ctrl+C basin.\n";
    std::cout << "[main] Yayin: rtsp://<board-ip>:8557/out\n";

    // Performans ozet dongusu: 1 saniyede bir decode/end-to-end FPS'i
    // frames_decoded/frames_processed'in ONCEKI olcumden bu yana artisindan
    // hesaplar (kesin gecen sureye bolerek — sleep_for'un tam 1000ms
    // surmemesi olasiligina karsi wall-clock kullanilir).
    uint64_t prev_decoded = 0;
    uint64_t prev_processed = 0;
    auto prev_t = std::chrono::steady_clock::now();

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        auto stats = orchestrator.getStats();
        auto now_t = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now_t - prev_t).count();
        if (elapsed_s < 1e-3) elapsed_s = 1e-3;

        double decode_fps = static_cast<double>(stats.frames_decoded - prev_decoded) / elapsed_s;
        double end_to_end_fps =
            static_cast<double>(stats.frames_processed - prev_processed) / elapsed_s;

        prev_decoded = stats.frames_decoded;
        prev_processed = stats.frames_processed;
        prev_t = now_t;

        std::cout << "[main] decode=" << stats.frames_decoded
                  << " (" << stats.decode_ms << "ms, " << decode_fps << " fps)"
                  << " letterbox=" << stats.letterbox_ms << "ms"
                  << " inference=" << stats.inference_ms << "ms"
                  << " postprocess=" << stats.postprocess_ms << "ms"
                  << " cizim=" << stats.draw_ms << "ms"
                  << " islenen=" << stats.frames_processed
                  << " (end-to-end " << end_to_end_fps << " fps)"
                  << " dusen(bayat)=" << stats.frames_dropped
                  << " dusen(npu->post)=" << stats.npu_results_dropped
                  << " aktif_track=" << stats.active_tracks
                  << " kaybolan=" << stats.lost_tracks << "\n";
    }

    std::cout << "[main] Pipeline durduruluyor...\n";

    orchestrator.stop();
    streamer.stop();

    std::cout << "[main] Pipeline kapatildi.\n";
    return 0;
}
