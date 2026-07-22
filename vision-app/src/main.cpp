#include "ingestion/RtspDemuxer.hpp"
#include "decode/MppDecoder.hpp"
#include "streaming/RtspStreamer.hpp"

#include <csignal>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

static std::atomic<bool> g_running{true};

void signal_handler(int signum) {
    (void)signum;
    g_running.store(false);
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string rtsp_url = "rtsp://localhost:8554/video";
    if (argc > 1) {
        rtsp_url = argv[1];
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  MyApp RTSP Demux + Decode + Stream" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input RTSP URL: " << rtsp_url << std::endl;
    std::cout << "Pipeline: Input Demux -> Decoder -> NV12 Stream -> Output RTSP" << std::endl;
    std::cout << "Çıkmak için Ctrl+C basin." << std::endl;
    std::cout << "========================================" << std::endl;

    MppDecoder decoder(VideoCoding::H264);
    
    RtspStreamer* streamer = nullptr;

    decoder.setFrameCallback([&streamer](DmaBufferPtr frame) {
        static int frame_count = 0;
        frame_count++;
        
        if (!streamer && frame) {
            std::cout << "[Main] İlk frame: " << frame->width << "x" << frame->height
                      << " - Streamer başlatılıyor...\n";
            streamer = new RtspStreamer(8554, "/live", frame->width, frame->height, 30);
            if (streamer->start()) {
                std::cout << "[Main] RTSP yayını: rtsp://<board-ip>:8554/live\n";
            } else {
                std::cerr << "[Main] Streamer başlatılamadı!\n";
                delete streamer;
                streamer = nullptr;
            }
        }
        
        if (streamer && frame) {
            streamer->pushRawNV12(frame);
        }
        
        if (frame_count % 30 == 0) {
            std::cout << "[Frame #" << frame_count
                      << "] " << frame->width << "x" << frame->height
                      << " pts=" << frame->pts_ns << "ns\n";
        }
    });

    RtspDemuxer demuxer(rtsp_url, 100);

    demuxer.setPacketCallback([&decoder](EncodedPacket&& packet) {
        decoder.feedPacket(std::move(packet));
    });

    if (!decoder.start()) {
        std::cerr << "[ERROR] Decoder başlatılamadı!" << std::endl;
        return 1;
    }

    if (!demuxer.start()) {
        std::cerr << "[ERROR] Demuxer başlatılamadı!" << std::endl;
        return 1;
    }

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

            std::cout << "[Stats] " << (df / elapsed) << " FPS decode, "
                      << "drop: " << dstats.frames_dropped
                      << " pkt_fail: " << dstats.packets_failed
                      << " demux_drop: " << mstats.packets_dropped << "\n";

            last_report = now;
            last_frames = dstats.frames_decoded;
        }
    }

    if (streamer) {
        streamer->stop();
        delete streamer;
    }
    
    demuxer.stop();
    decoder.stop();
    std::cout << "[EXIT] Pipeline durduruldu." << std::endl;

    return 0;
}