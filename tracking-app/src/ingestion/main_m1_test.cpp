// M1 doğrulama: SADECE RTSP demux hızını/kararlılığını ölçer.
// Henüz decode/RGA/NPU yok — amaç ağ+demux katmanının sağlam olduğunu
// kanıtlamak. M2'de bu paketler MPP decoder'a verilecek.
//
// Derleme:
//   g++ -std=c++17 -O2 \
//     src/ingestion/RtspDemuxer.cpp src/ingestion/main_m1_test.cpp \
//     -Iinclude \
//     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0) \
//     -lpthread -o m1_rtsp_test
//
// Çalıştırma:
//   ./m1_rtsp_test rtsp://kullanici:sifre@ip:port/yol

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
        std::cerr << "Kullanım: " << argv[0] << " rtsp://.../yol [latency_ms]\n";
        return 1;
    }

    uint32_t latency_ms = (argc >= 3) ? static_cast<uint32_t>(std::stoul(argv[2])) : 100;

    std::signal(SIGINT, onSigint);

    RtspDemuxer demuxer(argv[1], latency_ms);

    std::atomic<uint64_t> last_packet_size{0};
    demuxer.setPacketCallback([&](EncodedPacket&& pkt) {
        last_packet_size.store(pkt.data.size(), std::memory_order_relaxed);
        // M1'de bilerek başka bir şey yapmıyoruz.
    });

    if (!demuxer.start()) {
        std::cerr << "Demuxer başlatılamadı.\n";
        return 1;
    }

    std::cout << "RTSP demux çalışıyor (latency=" << latency_ms
              << "ms), Ctrl+C ile durdurun...\n";

    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_packets = 0;
    uint64_t last_bytes = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            auto stats = demuxer.getStats();
            uint64_t dp = stats.packets_received - last_packets;
            uint64_t db = stats.bytes_received - last_bytes;

            std::cout << "[M1] " << (dp / elapsed) << " paket/sn, "
                      << (db / elapsed / 1024.0) << " KB/sn, "
                      << "toplam düşen: " << stats.packets_dropped
                      << ", son paket: " << last_packet_size.load() << " byte\n";

            last_report = now;
            last_packets = stats.packets_received;
            last_bytes = stats.bytes_received;
        }
    }

    demuxer.stop();
    std::cout << "Durduruldu. Toplam istatistik:\n";
    auto final_stats = demuxer.getStats();
    std::cout << "  Paket: " << final_stats.packets_received
              << ", Bayt: " << final_stats.bytes_received
              << ", Düşen: " << final_stats.packets_dropped << "\n";
    return 0;
}
