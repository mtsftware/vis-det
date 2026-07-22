// M11: Faz 1 uçtan uca kablolama, TCP/gRPC üzerinden — CLAUDE.md "PHASE 1:
// Board Application".
//
// M10'un birebir aynısı (PipelineOrchestrator, çizim, RTSP çıkışı) — tek
// fark: BBOX alımı (UART/MAVLINK) yerine GrpcBboxSubscriber kullanılıyor.
// Board gRPC CLIENT'tır, GCS/yer istasyonu gRPC SERVER'dır — board GCS'e
// bağlanır, Stream (server-streaming) RPC'si ile abone olur, BBox OKUR.
// TargetPoint gönderimi (M10'daki UDP/MAVLINK)
// ŞİMDİLİK KALDIRILDI — akış artık tek yönlü (GCS -> board). BboxCallback
// sözleşmesi UartMavlinkLink ile birebir aynı olduğu için
// PipelineOrchestrator'a hiç dokunulmadı.
//
// Derleme (önce proto üretimi gerekir — bkz. aşağıdaki protoc adımı;
// -I /usr/include, google/protobuf/empty.proto'yu bulmak için gerekli):
//   protoc -I proto -I /usr/include --cpp_out=generated --grpc_out=generated \
//     --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/tracking.proto
//
//   g++ -std=c++14 -O2 \
//     src/ingestion/RtspDemuxer.cpp src/decode/MppDecoder.cpp \
//     src/buffer/DmaBufferPool.cpp src/preprocess/RgaPreprocessor.cpp \
//     src/inference/RknnInferenceEngine.cpp \
//     src/tracking/LightTrackImpl.cpp src/tracking/NanoTrackImpl.cpp \
//     src/streaming/RtspStreamer.cpp src/pipeline/PipelineOrchestrator.cpp \
//     src/comm/GrpcBboxSubscriber.cpp \
//     generated/tracking.pb.cc generated/tracking.grpc.pb.cc \
//     src/pipeline/main_m11_test.cpp \
//     -Iinclude -Igenerated -Ilibrknn_api/include -I/usr/include/rockchip -I/usr/include/rga \
//     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 \
//        gstreamer-rtsp-server-1.0 grpc++ protobuf) \
//     -Llibrknn_api/aarch64 -lrknnrt -Wl,-rpath,librknn_api/aarch64 \
//     -lrockchip_mpp -lrga -lpthread -o m11_pipeline_test
//
// Çalıştırma (kartta):
//   ./m11_pipeline_test rtsp://<kaynak_ip>:8554/stream \
//       --model lighttrack --gcs-host 192.168.1.50 --gcs-port 50051
//
// İzleme (PC'de):
//   ffplay rtsp://<kart_ip>:8557/out

#include "comm/GrpcBboxSubscriber.hpp"
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
#include <cstdlib>
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
constexpr uint16_t kDefaultGrpcPort = 50051;

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

// M8/M9/M10 ile aynı çizim (imfillArray, tüm rect'ler 2-hizalı).
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
        std::cerr << "[M11] drawBboxOutline: imfillArray başarısız: " << imStrError(r)
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

// rtsp://host:port/yol -> "host". --gcs-host verilmezse GCS adresi olarak
// RTSP kaynağının host'u varsayılır (M10'daki --udp-host varsayılanıyla
// aynı mantık).
std::string hostFromRtspUrl(const std::string& url) {
    const std::string prefix = "rtsp://";
    size_t start = (url.compare(0, prefix.size(), prefix) == 0) ? prefix.size() : 0;
    size_t end = url.find_first_of(":/", start);
    if (end == std::string::npos) end = url.size();
    return url.substr(start, end - start);
}

enum class ModelKind { LightTrackFp16, NanoTrackInt8 };

void printUsage(const char* argv0) {
    std::cerr
        << "Kullanım: " << argv0
        << " rtsp://kaynak/yol [--model lighttrack|nanotrack] "
           "[--gcs-host HOST] [--gcs-port 50051] "
           "[cikis_port=8557] [width=1280] [height=720] "
           "[template.rknn] [search.rknn] [head.rknn]\n"
           "  --model lighttrack (varsayılan): FP16 LightTrack\n"
           "  --model nanotrack: INT8 NanoTrack\n"
           "  --gcs-host: gRPC/TCP server adresi (varsayılan: RTSP kaynağının host'u)\n"
           "  --gcs-port: gRPC/TCP server portu (varsayılan 50051)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    ModelKind model_kind = ModelKind::LightTrackFp16;
    std::string gcs_host;
    uint16_t gcs_port = kDefaultGrpcPort;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "[M11] " << flag << " bir değer bekliyor\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            std::string val = needValue("--model");
            if (val == "lighttrack") {
                model_kind = ModelKind::LightTrackFp16;
            } else if (val == "nanotrack") {
                model_kind = ModelKind::NanoTrackInt8;
            } else {
                std::cerr << "[M11] Bilinmeyen --model değeri: " << val << "\n";
                return 1;
            }
        } else if (arg == "--gcs-host") {
            gcs_host = needValue("--gcs-host");
        } else if (arg == "--gcs-port") {
            gcs_port = static_cast<uint16_t>(std::stoi(needValue("--gcs-port")));
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    std::string rtsp_url = positional[0];
    if (gcs_host.empty()) gcs_host = hostFromRtspUrl(rtsp_url);

    uint16_t out_port =
        (positional.size() >= 2) ? static_cast<uint16_t>(std::stoi(positional[1])) : 8557;
    uint32_t out_w =
        (positional.size() >= 3) ? static_cast<uint32_t>(std::stoi(positional[2])) : 1280;
    uint32_t out_h =
        (positional.size() >= 4) ? static_cast<uint32_t>(std::stoi(positional[3])) : 720;

    std::string template_model, search_model, head_model;
    if (model_kind == ModelKind::LightTrackFp16) {
        template_model = (positional.size() >= 5) ? positional[4] : "models/lighttrack_init_fp16.rknn";
        search_model = (positional.size() >= 6) ? positional[5] : "models/lighttrack_backbone_fp16.rknn";
        head_model = (positional.size() >= 7) ? positional[6] : "models/lighttrack_neck_head_fp16.rknn";
    } else {
        template_model = (positional.size() >= 5) ? positional[4] : "models/T_model_backbone.rknn";
        search_model = (positional.size() >= 6) ? positional[5] : "models/X_model_backbone.rknn";
        head_model = (positional.size() >= 7) ? positional[6] : "models/model_head.rknn";
    }

    std::signal(SIGINT, onSigint);

    // CLAUDE.md kural #6: RKNN iç işçileri affinity'yi rknn_init anında
    // miras alır — model yüklemesi pin/restore arasında. İnference node
    // (tracker) burada, pipeline başlamadan ÖNCE yükleniyor — Faz 1'in
    // "runtime'da dinamik ekleme/çıkarma yok" kuralı bunu gerektiriyor.
    PipelineOrchestrator::pinCurrentThreadToBigCores();

    std::unique_ptr<LightTrackImpl> lighttrack;
    std::unique_ptr<NanoTrackImpl> nanotrack;
    ITracker* tracker = nullptr;

    if (model_kind == ModelKind::LightTrackFp16) {
        lighttrack = std::make_unique<LightTrackImpl>();
        if (!lighttrack->loadModels(template_model, search_model, head_model)) {
            std::cerr << "[M11] LightTrack modelleri yüklenemedi\n";
            return 1;
        }
        tracker = lighttrack.get();
        std::cout << "[M11] Tracker: LightTrack (FP16) — bypass modunda hazır\n";
    } else {
        nanotrack = std::make_unique<NanoTrackImpl>();
        if (!nanotrack->loadModels(template_model, search_model, head_model)) {
            std::cerr << "[M11] NanoTrack modelleri yüklenemedi\n";
            return 1;
        }
        tracker = nanotrack.get();
        std::cout << "[M11] Tracker: NanoTrack (INT8) — bypass modunda hazır\n";
    }

    PipelineOrchestrator::restoreCurrentThreadAllCores();

    RtspStreamer streamer(out_port, "/out", out_w, out_h, 30);
    if (!streamer.start()) {
        std::cerr << "[M11] RTSP streamer başlatılamadı\n";
        return 1;
    }

    GrpcBboxSubscriber grpc_link(gcs_host, gcs_port);
    if (!grpc_link.start()) {
        std::cerr << "[M11] gRPC BBox aboneliği başlatılamadı\n";
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
            bool lost = (r.state == TrackerState::Lost);

            // Target Point gönderimi ŞİMDİLİK KALDIRILDI (proto artık tek
            // yönlü GCS -> board akışı tanımlıyor, bkz. dosya başı notu).

            // KRİTİK: bbox ASLA r.frame (MPP decode çıktısı) üzerine
            // YERİNDE çizilmez (bkz. M8/M9 notu — RgaPreprocessor::cloneFrame
            // header'ı). Çizim HER ZAMAN bağımsız bir RGA kopyası üzerinde.
            DmaBufferPtr draw_frame;
            if (output_copier.cloneFrame(r.frame, toRgaPixelFormat(r.frame->format),
                                          draw_frame)) {
                drawBboxOutline(draw_frame, r.track.bbox,
                                 colorForConfidence(r.track.confidence, lost));
                streamer.pushFrame(draw_frame);
                return;
            }
            std::cerr << "[M11] cloneFrame başarısız — bu karede çizim atlandı, "
                         "ham kare yayınlanıyor\n";
        }
        streamer.pushFrame(r.frame);
    });

    orchestrator.setLostCallback([](uint32_t consecutive) {
        if (consecutive == 1 || consecutive % 150 == 0) {
            std::cout << "[M11] hedef kayıp sinyali (ardışık " << consecutive
                      << " kare) — detector entegrasyonu burada devreye girecek\n";
        }
    });

    // Faz 1 §3: gRPC/TCP BBox aboneliği. BBox geldiğinde tracker'ı
    // etkinleştirir (bypass -> active) — inference node zaten pipeline'da
    // yüklü, sadece bir sonraki karede initialize() tetiklenir.
    grpc_link.setBboxCallback([&](const BBox& bbox) {
        std::cout << "[M11] BBox etkinleştiriliyor -> requestInitialize\n";
        orchestrator.requestInitialize(bbox);
    });

    // İlk hedef CLI'dan DEĞİL, gRPC stream'inden gelir — pipeline
    // idle/pass-through modda başlar (CLAUDE.md Faz 1 §2
    // "Pass-Through Architecture").
    if (!orchestrator.start()) {
        std::cerr << "[M11] Orkestratör başlatılamadı\n";
        return 1;
    }

    std::cout << "[M11] Pipeline çalışıyor (bypass), BBox bekleniyor, Ctrl+C ile "
                 "durdurun...\n";

    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_tracked = 0;
    uint64_t last_decoded = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            auto st = orchestrator.getStats();
            auto grpc_st = grpc_link.getStats();
            std::cout << "[M11] " << ((st.frames_tracked - last_tracked) / elapsed)
                      << " FPS tracking, "
                      << ((st.frames_decoded - last_decoded) / elapsed)
                      << " FPS decode, son confidence=" << st.last_confidence
                      << ", düşürülen kare: " << st.frames_dropped
                      << ", toplam kayıp kare: " << st.frames_lost
                      << ", gRPC bbox alınan: " << grpc_st.bbox_received
                      << ", yeniden bağlanma: " << grpc_st.stream_reconnects << "\n";
            last_report = now;
            last_tracked = st.frames_tracked;
            last_decoded = st.frames_decoded;
        }
    }

    orchestrator.stop();
    grpc_link.stop();
    streamer.stop();
    tracker->shutdown();
    std::cout << "[M11] Durduruldu.\n";
    return 0;
}