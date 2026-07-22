// M2 doğrulama: RtspDemuxer (M1) + native MPP decode zincirlenmiş halde
// çalışır. Amaç: gerçek decode FPS'i ölçmek ve zero-copy DmaBuffer
// üretiminin (fd, stride, çözünürlük) doğru geldiğini teyit etmek.
// Henüz RGA/NPU yok — kareler decode edilir, loglanır, serbest bırakılır.
//
// Derleme:
//   g++ -std=c++14 -O2 \
//     src/ingestion/RtspDemuxer.cpp src/decode/MppDecoder.cpp \
//     src/decode/main_m2_test.cpp \
//     -Iinclude \
//     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 rockchip_mpp) \
//     -lpthread -o m2_decode_test
//
// Çalıştırma:
//   ./m2_decode_test rtsp://.../yol

#include "decode/MppDecoder.hpp"
#include "ingestion/RtspDemuxer.hpp"

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
        std::cerr << "Kullanım: " << argv[0] << " rtsp://.../yol\n";
        return 1;
    }

    std::signal(SIGINT, onSigint);

    RtspDemuxer demuxer(argv[1]);
    MppDecoder decoder(VideoCoding::H264);

    std::atomic<bool> logged_resolution{false};

    decoder.setFrameCallback([&](DmaBufferPtr frame) {
        if (!logged_resolution.exchange(true)) {
            std::cout << "[M2] İlk kare: " << frame->width << "x" << frame->height
                      << " (stride " << frame->w_stride << "x" << frame->h_stride
                      << "), fd=" << frame->fd << ", size=" << frame->size << " byte\n";
        }
        // M2'de bilerek başka bir şey yapmıyoruz — frame burada scope'tan
        // çıkınca release_fn (mpp_frame_deinit) tetiklenir. M3'te bu
        // noktada RGA'ya devredilecek.
    });

    demuxer.setPacketCallback(
        [&](EncodedPacket&& pkt) { decoder.feedPacket(std::move(pkt)); });

    if (!decoder.start()) {
        std::cerr << "MPP decoder başlatılamadı.\n";
        return 1;
    }
    if (!demuxer.start()) {
        std::cerr << "Demuxer başlatılamadı.\n";
        return 1;
    }

    std::cout << "RTSP demux + native MPP decode çalışıyor, Ctrl+C ile durdurun...\n";

    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_frames = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            auto dstats = decoder.getStats();
            auto mstats = demuxer.getStats();
            uint64_t df = dstats.frames_decoded - last_frames;

            std::cout << "[M2] " << (df / elapsed) << " FPS decode, "
                      << "düşen kare: " << dstats.frames_dropped
                      << ", başarısız paket: " << dstats.packets_failed
                      << ", demux düşen: " << mstats.packets_dropped << "\n";

            last_report = now;
            last_frames = dstats.frames_decoded;
        }
    }

    demuxer.stop();
    decoder.stop();
    std::cout << "Durduruldu.\n";
    return 0;
}
