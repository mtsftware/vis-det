// M9: LightTrack (FP16) / NanoTrack (INT8) arasında ÇALIŞMA-ZAMANI geçiş.
// M8'in birebir aynısı (PipelineOrchestrator kablolaması, çizim, RTSP
// çıkışı) — tek fark: --model lighttrack|nanotrack argümanına göre hangi
// ITracker somutunun (LightTrackImpl ya da NanoTrackImpl) yükleneceğine
// karar veriyor. İkisi de aynı ITracker arayüzünü karşıladığı için
// PipelineOrchestrator hiçbir değişiklik gerektirmedi (05-ITracker.md'nin
// tasarım amacı tam olarak buydu).
//
// Derleme:
//   g++ -std=c++14 -O2 \
//     src/ingestion/RtspDemuxer.cpp src/decode/MppDecoder.cpp \
//     src/buffer/DmaBufferPool.cpp src/preprocess/RgaPreprocessor.cpp \
//     src/inference/RknnInferenceEngine.cpp \
//     src/tracking/LightTrackImpl.cpp src/tracking/NanoTrackImpl.cpp \
//     src/streaming/RtspStreamer.cpp src/pipeline/PipelineOrchestrator.cpp \
//     src/pipeline/main_m9_test.cpp \
//     -Iinclude -Ilibrknn_api/include -I/usr/include/rockchip -I/usr/include/rga \
//     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 \
//        gstreamer-rtsp-server-1.0) \
//     -Llibrknn_api/aarch64 -lrknnrt -Wl,-rpath,librknn_api/aarch64 \
//     -lrockchip_mpp -lrga -lpthread -o m9_pipeline_test
//
// Çalıştırma (kartta):
//   ./m9_pipeline_test rtsp://<kaynak_ip>:8554/stream 233,152,109,222 \
//       --model lighttrack
//   ./m9_pipeline_test rtsp://<kaynak_ip>:8554/stream 233,152,109,222 \
//       --model nanotrack 8557 1280 720 \
//       models/T_model_backbone.rknn models/X_model_backbone.rknn models/model_head.rknn
//
// İzleme (PC'de):
//   ffplay rtsp://<kart_ip>:8557/out

#include "pipeline/PipelineOrchestrator.hpp"
#include "preprocess/RgaPreprocessor.hpp"
#include "streaming/RtspStreamer.hpp"
#include "tracking/LightTrackImpl.hpp"
#include "tracking/NanoTrackImpl.hpp"

#include "im2d.h"
#include "rga.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_running{true};
void onSigint(int) { g_running.store(false); }

constexpr float kScoreGreenThreshold = 0.7f;
constexpr float kScoreOrangeThreshold = 0.4f;
constexpr int kBoxThickness = 4;

int toRgaFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::NV12:
            return RK_FORMAT_YCbCr_420_SP;
        case PixelFormat::NV16:
            return RK_FORMAT_YCbCr_422_SP;
        case PixelFormat::RGB888:
            return RK_FORMAT_RGB_888;
        case PixelFormat::BGR888:
            return RK_FORMAT_BGR_888;
        default:
            return RK_FORMAT_YCbCr_420_SP;
    }
}

RgaPixelFormat toRgaPixelFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::NV12:
            return RgaPixelFormat::NV12;
        case PixelFormat::NV16:
            return RgaPixelFormat::NV16;
        case PixelFormat::RGB888:
            return RgaPixelFormat::RGB888;
        case PixelFormat::BGR888:
            return RgaPixelFormat::BGR888;
        default:
            return RgaPixelFormat::NV12;
    }
}

// M8 ile aynı çizim (imfillArray, tüm rect'ler 2-hizalı).
void drawBboxOutline(const DmaBufferPtr& frame, const BBox& bbox, int color) {
    static bool logged_error = false;
    if (!frame || frame->fd < 0) return;

    int fw = static_cast<int>(frame->width);
    int fh = static_cast<int>(frame->height);

    int x0 = std::max(0, bbox.x);
    int y0 = std::max(0, bbox.y);
    int x1 = std::min(fw, bbox.x + bbox.width);
    int y1 = std::min(fh, bbox.y + bbox.height);
    if (x1 <= x0 || y1 <= y0) return;

    int t = std::min(kBoxThickness, std::min(x1 - x0, y1 - y0) / 2);
    if (t < 1) t = 1;

    rga_buffer_t buf = wrapbuffer_fd(frame->fd, fw, fh, toRgaFormat(frame->format),
                                      static_cast<int>(frame->w_stride),
                                      static_cast<int>(frame->h_stride));

    auto alignRectEven = [](im_rect r) {
        r.x -= (r.x & 1);
        r.y -= (r.y & 1);
        r.width -= (r.width & 1);
        r.height -= (r.height & 1);
        if (r.width < 2) r.width = 2;
        if (r.height < 2) r.height = 2;
        return r;
    };

    im_rect rects[4] = {
        alignRectEven(im_rect{x0, y0, x1 - x0, t}),
        alignRectEven(im_rect{x0, std::max(y0, y1 - t), x1 - x0, t}),
        alignRectEven(im_rect{x0, y0, t, y1 - y0}),
        alignRectEven(im_rect{std::max(x0, x1 - t), y0, t, y1 - y0}),
    };

    IM_STATUS r = imfillArray(buf, rects, 4, static_cast<uint32_t>(color));
    if (!logged_error && r != IM_STATUS_SUCCESS) {
        std::cerr << "[M9] drawBboxOutline: imfillArray başarısız: " << imStrError(r)
                  << " (bir kez loglanıyor)\n";
        logged_error = true;
    }
}

int colorForConfidence(float confidence, bool lost) {
    if (lost || confidence <= kScoreOrangeThreshold) {
        return (255 << 16) | (0 << 8) | 0;  // kırmızı
    }
    if (confidence <= kScoreGreenThreshold) {
        return (255 << 16) | (165 << 8) | 0;  // turuncu
    }
    return (0 << 16) | (255 << 8) | 0;  // yeşil
}

bool parseBbox(const std::string& s, BBox& out) {
    std::stringstream ss(s);
    std::string tok;
    int vals[4];
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try {
            vals[i] = std::stoi(tok);
        } catch (...) {
            return false;
        }
    }
    out.x = vals[0];
    out.y = vals[1];
    out.width = vals[2];
    out.height = vals[3];
    return out.width > 0 && out.height > 0;
}

enum class ModelKind { LightTrackFp16, NanoTrackInt8 };

void printUsage(const char* argv0) {
    std::cerr << "Kullanım: " << argv0
              << " rtsp://kaynak/yol x,y,w,h [--model lighttrack|nanotrack] "
                 "[cikis_port=8557] [width=1280] [height=720] "
                 "[template.rknn] [search.rknn] [head.rknn]\n"
                 "  --model lighttrack (varsayılan): FP16 LightTrack, "
                 "varsayılan yollar models/lighttrack_{init,backbone,neck_head}_fp16.rknn\n"
                 "  --model nanotrack: INT8 NanoTrack, varsayılan yollar "
                 "models/{T_model_backbone,X_model_backbone,model_head}.rknn\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    // --model flag'ini pozisyonel argümanlardan ayıkla; kalan argümanlar
    // M8'deki SIRA ile aynı şekilde işlenir (geriye dönük CLI uyumu).
    ModelKind model_kind = ModelKind::LightTrackFp16;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "[M9] --model bir değer bekliyor (lighttrack|nanotrack)\n";
                return 1;
            }
            std::string val = argv[++i];
            if (val == "lighttrack") {
                model_kind = ModelKind::LightTrackFp16;
            } else if (val == "nanotrack") {
                model_kind = ModelKind::NanoTrackInt8;
            } else {
                std::cerr << "[M9] Bilinmeyen --model değeri: " << val
                          << " (beklenen: lighttrack|nanotrack)\n";
                return 1;
            }
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string rtsp_url = positional[0];
    BBox init_bbox;
    if (!parseBbox(positional[1], init_bbox)) {
        std::cerr << "[M9] Geçersiz bbox formatı, beklenen: x,y,w,h (pozitif w/h)\n";
        return 1;
    }

    uint16_t out_port = (positional.size() >= 3) ? static_cast<uint16_t>(std::stoi(positional[2])) : 8557;
    uint32_t out_w = (positional.size() >= 4) ? static_cast<uint32_t>(std::stoi(positional[3])) : 1280;
    uint32_t out_h = (positional.size() >= 5) ? static_cast<uint32_t>(std::stoi(positional[4])) : 720;

    std::string template_model, search_model, head_model;
    if (model_kind == ModelKind::LightTrackFp16) {
        template_model = (positional.size() >= 6) ? positional[5] : "models/lighttrack_init_fp16.rknn";
        search_model = (positional.size() >= 7) ? positional[6] : "models/lighttrack_backbone_fp16.rknn";
        head_model = (positional.size() >= 8) ? positional[7] : "models/lighttrack_neck_head_fp16.rknn";
    } else {
        template_model = (positional.size() >= 6) ? positional[5] : "models/T_model_backbone.rknn";
        search_model = (positional.size() >= 7) ? positional[6] : "models/X_model_backbone.rknn";
        head_model = (positional.size() >= 8) ? positional[7] : "models/model_head.rknn";
    }

    std::signal(SIGINT, onSigint);

    // CLAUDE.md kural #6: RKNN iç işçileri affinity'yi rknn_init anında
    // miras alır — model yüklemesi pin/restore arasında.
    PipelineOrchestrator::pinCurrentThreadToBigCores();

    // Sadece SEÇİLEN tracker somutlaştırılır — ITracker& üzerinden
    // PipelineOrchestrator'a verilir, ikisi asla aynı anda yüklenmez.
    std::unique_ptr<LightTrackImpl> lighttrack;
    std::unique_ptr<NanoTrackImpl> nanotrack;
    ITracker* tracker = nullptr;

    if (model_kind == ModelKind::LightTrackFp16) {
        lighttrack = std::make_unique<LightTrackImpl>();
        if (!lighttrack->loadModels(template_model, search_model, head_model)) {
            std::cerr << "[M9] LightTrack modelleri yüklenemedi\n";
            return 1;
        }
        tracker = lighttrack.get();
        std::cout << "[M9] Tracker: LightTrack (FP16)\n";
    } else {
        nanotrack = std::make_unique<NanoTrackImpl>();
        if (!nanotrack->loadModels(template_model, search_model, head_model)) {
            std::cerr << "[M9] NanoTrack modelleri yüklenemedi\n";
            return 1;
        }
        tracker = nanotrack.get();
        std::cout << "[M9] Tracker: NanoTrack (INT8)\n";
    }

    PipelineOrchestrator::restoreCurrentThreadAllCores();

    RtspStreamer streamer(out_port, "/out", out_w, out_h, 30);
    if (!streamer.start()) {
        std::cerr << "[M9] RTSP streamer başlatılamadı\n";
        return 1;
    }

    PipelineConfig cfg;
    cfg.rtsp_url = rtsp_url;
    PipelineOrchestrator orchestrator(*tracker, cfg);

    // Çıktı katmanı için AYRI bir RgaPreprocessor — sadece frame kopyalama
    // amaçlı (tracker'ın kendi preprocessor_'ından TAMAMEN bağımsız).
    RgaPreprocessor output_copier;

    orchestrator.setResultCallback([&](const FrameResult& r) {
        if (r.tracked) {
            // KRİTİK: bbox ASLA r.frame (MPP decode çıktısı) üzerine
            // YERİNDE çizilmez — bkz. RgaPreprocessor::cloneFrame header
            // notu (kod çözücü bu belleği gelecekteki P-frame'ler için
            // referans kare olarak da kullanıyor olabilir; yerinde çizim
            // bunu bozup görüntüye kalıcı yapışan bir artefakt olarak
            // yansıyabilir — NanoTrack'in daha hızlı backbone'u sayesinde
            // çakışma penceresi daha sık tetiklendiği için canlı testte
            // fark edildi). Çizim HER ZAMAN bağımsız bir RGA kopyası
            // üzerinde yapılır.
            bool lost = (r.state == TrackerState::Lost);
            DmaBufferPtr draw_frame;
            if (output_copier.cloneFrame(r.frame, toRgaPixelFormat(r.frame->format),
                                          draw_frame)) {
                drawBboxOutline(draw_frame, r.track.bbox,
                                 colorForConfidence(r.track.confidence, lost));
                streamer.pushFrame(draw_frame);
                return;
            }
            std::cerr << "[M9] cloneFrame başarısız — bu karede çizim atlandı, "
                         "ham kare yayınlanıyor\n";
        }
        streamer.pushFrame(r.frame);
    });

    orchestrator.setLostCallback([](uint32_t consecutive) {
        if (consecutive == 1 || consecutive % 150 == 0) {
            std::cout << "[M9] hedef kayıp sinyali (ardışık " << consecutive
                      << " kare) — detector entegrasyonu burada devreye girecek\n";
        }
    });

    orchestrator.requestInitialize(init_bbox);

    if (!orchestrator.start()) {
        std::cerr << "[M9] Orkestratör başlatılamadı\n";
        return 1;
    }

    std::cout << "[M9] Pipeline çalışıyor, Ctrl+C ile durdurun...\n";

    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_tracked = 0;
    uint64_t last_decoded = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            auto st = orchestrator.getStats();
            std::cout << "[M9] " << ((st.frames_tracked - last_tracked) / elapsed)
                      << " FPS tracking, "
                      << ((st.frames_decoded - last_decoded) / elapsed)
                      << " FPS decode, son confidence=" << st.last_confidence
                      << ", düşürülen kare: " << st.frames_dropped
                      << ", toplam kayıp kare: " << st.frames_lost << "\n";
            last_report = now;
            last_tracked = st.frames_tracked;
            last_decoded = st.frames_decoded;
        }
    }

    orchestrator.stop();
    streamer.stop();
    tracker->shutdown();
    std::cout << "[M9] Durduruldu.\n";
    return 0;
}