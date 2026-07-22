#include "ingestion/RtspDemuxer.hpp"
#include "decode/MppDecoder.hpp"

#include <csignal>
#include <iostream>
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
    std::cout << "  MyApp RTSP Demux + Decode Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RTSP URL: " << rtsp_url << std::endl;
    std::cout << "Pipeline: Demuxer -> Decoder -> NV12 Frame Callback" << std::endl;
    std::cout << "Çıkmak için Ctrl+C basin." << std::endl;
    std::cout << "========================================" << std::endl;

    // Decoder'ı oluştur (varsayılan H.264)
    MppDecoder decoder(VideoCoding::H264);

    // Decoder callback: Her NV12 frame geldiğinde çağrılır
    decoder.setFrameCallback([](DmaBufferPtr frame) {
        static int frame_count = 0;
        frame_count++;
        std::cout << "[Frame #" << frame_count
                  << "] " << frame->width << "x" << frame->height
                  << " stride=" << frame->w_stride << "x" << frame->h_stride
                  << " fmt=" << (frame->format == PixelFormat::NV12 ? "NV12" : "OTHER")
                  << " pts=" << frame->pts_ns << "ns"
                  << " size=" << frame->size << "B" << std::endl;
    });

    // Demuxer'ı oluştur
    RtspDemuxer demuxer(rtsp_url, 100);

    // Demuxer → Decoder bağlantısı
    demuxer.setPacketCallback([&decoder](EncodedPacket&& packet) {
        decoder.feedPacket(std::move(packet));
    });

    // Başlat
    if (!decoder.start()) {
        std::cerr << "[ERROR] Decoder başlatılamadı!" << std::endl;
        return 1;
    }

    if (!demuxer.start()) {
        std::cerr << "[ERROR] Demuxer başlatılamadı!" << std::endl;
        return 1;
    }

    // FPS hesaplama döngüsü (tracking-app src/decode/main_m2_test.cpp referans)
    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_frames = 0;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            auto dstats = decoder.getStats();
            auto mstats = demuxer.getStats();
            uint64_t df = dstats.frames_decoded - last_frames;

            std::cout << "[Stats] " << (df / elapsed) << " FPS decode, "
                      << "düşen kare: " << dstats.frames_dropped
                      << ", başarısız paket: " << dstats.packets_failed
                      << ", demux düşen: " << mstats.packets_dropped << "\n";

            last_report = now;
            last_frames = dstats.frames_decoded;
        }
    }

    demuxer.stop();
    decoder.stop();
    std::cout << "[EXIT] Pipeline durduruldu." << std::endl;

    return 0;
}