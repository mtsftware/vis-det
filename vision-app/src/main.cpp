#include "ingestion/RtspDemuxer.hpp"

#include <csignal>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>

static volatile sig_atomic_t g_running = 1;

void signal_handler(int signum) {
    (void)signum;
    g_running = 0;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // RTSP URL'si argümandan al, yoksa varsayılan
    std::string rtsp_url = "rtsp://localhost:8554/video";
    if (argc > 1) {
        rtsp_url = argv[1];
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  MyApp RTSP Demuxer Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RTSP URL: " << rtsp_url << std::endl;
    std::cout << "Paket sayacini görmek için her 1 saniyede bir stats yazdirilacak." << std::endl;
    std::cout << "Çıkmak için Ctrl+C basin." << std::endl;
    std::cout << "========================================" << std::endl;

    RtspDemuxer demuxer(rtsp_url, 100);

    // Callback: her paket geldiğinde çağrılır
    demuxer.setPacketCallback([](EncodedPacket&& packet) {
        // Büyük paketleri göster (NAL üniteleri tipik olarak 100-15000 byte)
        if (packet.data.size() > 100) {
            std::cout << "[Packet] " << packet.data.size() 
                      << " bytes, PTS=" << packet.pts_ns << "ns" << std::endl;
        }
    });

    if (!demuxer.start()) {
        std::cerr << "[ERROR] Demuxer başlatılamadı!" << std::endl;
        return 1;
    }

    // Stats döngüsü
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        RtspDemuxer::Stats stats = demuxer.getStats();
        std::cout << "[Stats] recv=" << stats.packets_received
                  << " drop=" << stats.packets_dropped
                  << " bytes=" << stats.bytes_received << std::endl;
    }

    demuxer.stop();
    std::cout << "[EXIT] Demuxer durduruldu." << std::endl;

    return 0;
}