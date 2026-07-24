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
    config.model_path = "models/best-rk3588.rknn";
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
                      << result.tracked_objects.size() << " aktif track\n";
        }
    });

    if (!orchestrator.start()) {
        std::cerr << "[main] Pipeline baslatilamadi!\n";
        streamer.stop();
        return 1;
    }

    std::cout << "[main] Pipeline baslatildi. Cikmak icin Ctrl+C basin.\n";
    std::cout << "[main] Yayin: rtsp://<board-ip>:8557/out\n";

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        auto stats = orchestrator.getStats();
        std::cout << "[main] decode=" << stats.frames_decoded
                  << " islenen=" << stats.frames_processed
                  << " dusen(bayat)=" << stats.frames_dropped
                  << " aktif_track=" << stats.active_tracks
                  << " kaybolan=" << stats.lost_tracks << "\n";
    }

    std::cout << "[main] Pipeline durduruluyor...\n";

    orchestrator.stop();
    streamer.stop();

    std::cout << "[main] Pipeline kapatildi.\n";
    return 0;
}
