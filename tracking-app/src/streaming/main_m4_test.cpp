// M4 doğrulama: RTSP demux (M1) + native MPP decode (M2) + görsel
// doğrulama için RTSP yeniden yayın (appsrc -> mpph264enc -> GstRTSPServer).
//
// Amaç: decode edilen NV12 piksellerin GERÇEKTEN doğru olduğunu gözle
// görmek (M2 sadece hız/paket sayımı doğruladı, piksel doğruluğunu değil).
//
// Derleme:
//   g++ -std=c++14 -O2 \
//     src/ingestion/RtspDemuxer.cpp src/decode/MppDecoder.cpp \
//     src/streaming/RtspStreamer.cpp src/streaming/main_m4_test.cpp \
//     -Iinclude -I/usr/include/rockchip \
//     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 \
//        gstreamer-rtsp-server-1.0) \
//     -lrockchip_mpp -lpthread -o m4_stream_test
//
// Çalıştırma (kartta):
//   ./m4_stream_test rtsp://<webcam_pc_ip>:8554/stream 8555
//
// İzleme (PC'de, ayrı bir terminalde):
//   ffplay rtsp://<kart_ip>:8555/out

#include "decode/MppDecoder.hpp"
#include "ingestion/RtspDemuxer.hpp"
#include "streaming/RtspStreamer.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

namespace {
std::atomic<bool> g_running{true};
void onSigint(int) { g_running.store(false); }
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanım: " << argv[0]
                  << " rtsp://kaynak/yol [cikis_port=8555]\n";
        return 1;
    }

    uint16_t out_port = (argc >= 3) ? static_cast<uint16_t>(std::stoi(argv[2])) : 8555;

    std::signal(SIGINT, onSigint);

    RtspDemuxer demuxer(argv[1]);
    MppDecoder decoder(VideoCoding::H264);

    // Webcam testimizden bildiğimiz sabit çözünürlük (M2'deki "İlk kare:
    // 1280x720" logundan). Gerçek pipeline'da bu, decode'un ilk info-change
    // karesinden DİNAMİK alınmalı — burada M4'ü basit tutmak için sabit.
    RtspStreamer streamer(out_port, "/out", 1280, 720, 30);

    decoder.setFrameCallback([&](DmaBufferPtr frame) { streamer.pushFrame(frame); });

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

    std::cout << "Zincir çalışıyor (kaynak -> decode -> yeniden yayın), "
                 "Ctrl+C ile durdurun...\n";

    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_frames = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            auto dstats = decoder.getStats();
            uint64_t df = dstats.frames_decoded - last_frames;
            std::cout << "[M4] " << (df / elapsed) << " FPS decode, "
                      << "düşen kare: " << dstats.frames_dropped << "\n";
            last_report = now;
            last_frames = dstats.frames_decoded;
        }
    }

    demuxer.stop();
    decoder.stop();
    streamer.stop();
    std::cout << "Durduruldu.\n";
    return 0;
}
