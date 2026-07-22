// M3 doğrulama: RTSP demux + native MPP decode + RGA letterbox (imfill +
// improcess, immakeBorder DEĞİL) + görsel doğrulama için RTSP yeniden yayın.
//
// Amaç: RGA'nın aspect-ratio koruyarak letterbox yaptığını GÖZLE görmek —
// 1280x720 (16:9) kaynağın 640x640 (1:1) hedefe sığdırılırken üstte/altta
// gri kenarlık bırakması, görüntünün gerilmemiş/bozulmamış olması.
//
// KRİTİK: Kaynak formatı (NV12/NV16) MppDecoder'ın gerçekten tespit ettiği
// değerden okunuyor, VARSAYILMIYOR (bkz. DmaBuffer::format, M4'teki NV16
// keşfinden sonra düzeltildi).
//
// Derleme (M7'den itibaren src/buffer/DmaBufferPool.cpp de gerekli —
// RgaPreprocessor buffer'larını artık IBufferPool'dan alıyor):
//   g++ -std=c++14 -O2 \
//     src/ingestion/RtspDemuxer.cpp src/decode/MppDecoder.cpp \
//     src/buffer/DmaBufferPool.cpp src/preprocess/RgaPreprocessor.cpp \
//     src/streaming/RtspStreamer.cpp src/streaming/main_m3_test.cpp \
//     -Iinclude -I/usr/include/rockchip -I/usr/include/rga \
//     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 \
//        gstreamer-rtsp-server-1.0) \
//     -lrockchip_mpp -lrga -lpthread -o m3_rga_test
//
// Çalıştırma (kartta):
//   ./m3_rga_test rtsp://<webcam_pc_ip>:8554/stream 8556
//
// İzleme (PC'de):
//   ffplay rtsp://<kart_ip>:8556/out

#include "decode/MppDecoder.hpp"
#include "ingestion/RtspDemuxer.hpp"
#include "preprocess/RgaPreprocessor.hpp"
#include "streaming/RtspStreamer.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

namespace {
std::atomic<bool> g_running{true};
void onSigint(int) { g_running.store(false); }

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
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanım: " << argv[0]
                  << " rtsp://kaynak/yol [cikis_port=8556]\n";
        return 1;
    }

    uint16_t out_port = (argc >= 3) ? static_cast<uint16_t>(std::stoi(argv[2])) : 8556;

    std::signal(SIGINT, onSigint);

    RtspDemuxer demuxer(argv[1]);
    MppDecoder decoder(VideoCoding::H264);

    RgaPreprocessor preprocessor;
    preprocessor.configure(640, 640, RgaPixelFormat::NV12);

    // PC'ye giden stream ARTIK ORİJİNAL boyutta (webcam testimizden bilinen
    // 1280x720 — gerçek kamerada bu, decode'un ilk info-change karesinden
    // dinamik alınmalı). RGA letterbox işlemi ARKA PLANDA hâlâ çalışıyor
    // (performans ölçümünü sürdürmek için) ama sonucu artık PC'ye
    // GÖNDERİLMİYOR — sadece zamanlama/kararlılık doğrulaması için üretiliyor.
    RtspStreamer streamer(out_port, "/out", 1280, 720, 30);

    std::atomic<bool> logged_first{false};
    std::atomic<uint64_t> rga_frames{0};
    std::atomic<uint64_t> rga_total_us{0};
    std::atomic<uint64_t> rga_last_us{0};

    decoder.setFrameCallback([&](DmaBufferPtr frame) {
        DmaBufferPtr rga_out;
        LetterboxResult transform;

        RgaPixelFormat src_fmt = toRgaPixelFormat(frame->format);

        auto t0 = std::chrono::steady_clock::now();
        bool ok = preprocessor.process(frame, src_fmt, rga_out, transform);
        auto t1 = std::chrono::steady_clock::now();

        if (!ok) {
            std::cerr << "[M3] RGA process başarısız\n";
            return;
        }

        uint64_t us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        rga_frames.fetch_add(1, std::memory_order_relaxed);
        rga_total_us.fetch_add(us, std::memory_order_relaxed);
        rga_last_us.store(us, std::memory_order_relaxed);

        if (!logged_first.exchange(true)) {
            std::cout << "[M3] Letterbox: ratio=" << transform.ratio
                      << " offset=(" << transform.dst_offset_x << ","
                      << transform.dst_offset_y << ") scaled="
                      << transform.scaled_width << "x" << transform.scaled_height
                      << "\n";
        }

        streamer.pushFrame(frame);  // ORİJİNAL decode çıktısı (NV16, 1280x720)
    });

    demuxer.setPacketCallback(
        [&](EncodedPacket&& pkt) { decoder.feedPacket(std::move(pkt)); });

    if (!streamer.start()) {
        std::cerr << "RTSP streamer başlatılamadı.\n";
        return 1;
    }
    if (!decoder.start()) {
        std::cerr << "MPP decoder başlatılamadı.\n";
        return 1;
    }
    if (!demuxer.start()) {
        std::cerr << "Demuxer başlatılamadı.\n";
        return 1;
    }

    std::cout << "Zincir çalışıyor (kaynak -> decode -> RGA letterbox -> "
                 "yeniden yayın), Ctrl+C ile durdurun...\n";

    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_frames = 0;
    uint64_t last_rga_frames = 0;
    uint64_t last_rga_total_us = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            auto dstats = decoder.getStats();
            uint64_t df = dstats.frames_decoded - last_frames;

            uint64_t rf = rga_frames.load(std::memory_order_relaxed);
            uint64_t rt = rga_total_us.load(std::memory_order_relaxed);
            uint64_t d_rf = rf - last_rga_frames;
            uint64_t d_rt = rt - last_rga_total_us;
            double avg_rga_us = (d_rf > 0) ? static_cast<double>(d_rt) / d_rf : 0.0;

            std::cout << "[M3] " << (df / elapsed) << " FPS decode, "
                      << "RGA ortalama=" << avg_rga_us << "us "
                      << "son=" << rga_last_us.load(std::memory_order_relaxed)
                      << "us, düşen kare: " << dstats.frames_dropped << "\n";

            last_report = now;
            last_frames = dstats.frames_decoded;
            last_rga_frames = rf;
            last_rga_total_us = rt;
        }
    }

    demuxer.stop();
    decoder.stop();
    streamer.stop();
    std::cout << "Durduruldu.\n";
    return 0;
}