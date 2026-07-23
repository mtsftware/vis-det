#include "ingestion/RtspDemuxer.hpp"
#include "decode/MppDecoder.hpp"
#include "streaming/RtspStreamer.hpp"
#include "rga_processing/RgaPreprocessor.hpp"
#include "types/DmaBuffer.hpp"
#include "types/VideoCoding.hpp"

#include <csignal>
#include <iostream>
#include <mutex>
#include <atomic>
#include <string>
#include <thread>

static std::atomic<bool> g_running{true};
static std::mutex g_frame_mutex;

void signalHandler(int /*signum*/) {
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::string rtsp_url = "rtsp://localhost:8554/video";
    if (argc > 1) {
        rtsp_url = argv[1];
    }

    std::cout << "============================================================\n";
    std::cout << "  MyApp - Phase 4: RGA On Isleme Pipeline (Renkli DMA)\n";
    std::cout << "============================================================\n";
    std::cout << "RTSP Kaynak : " << rtsp_url << "\n";
    std::cout << "Yayin URL   : rtsp://<board-ip>:8554/live\n";
    std::cout << "============================================================\n";

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    const uint32_t target_w = 640;
    const uint32_t target_h = 640;
    const uint32_t fps = 30;

    // 1. RTSP Demuxer baslat
    RtspDemuxer demuxer(rtsp_url, 5000);
    if (!demuxer.start()) {
        std::cerr << "[main] Demuxer baslatilamadi!\n";
        return 1;
    }
    std::cout << "[main] Demuxer basarili.\n";

    // 2. MPP Decoder baslat
    MppDecoder decoder(VideoCoding::H264);
    if (!decoder.start()) {
        std::cerr << "[main] Decoder baslatilamadi!\n";
        return 1;
    }
    std::cout << "[main] Decoder basarili.\n";

    uint32_t original_width = 1280;
    uint32_t original_height = 720;
    PixelFormat detected_format = PixelFormat::Unknown;
    std::atomic<int> frame_count{0};

    // 3. RTSP Streamer baslat — orijinal boyutlarla (format sonradan ayarlanacak)
    RtspStreamer streamer(8554, "/live", original_width, original_height, fps);
    if (!streamer.start()) {
        std::cerr << "[main] Streamer baslatilamadi!\n";
        return 1;
    }
    std::cout << "[main] Streamer basarili (NV12/NV16).\n";

    // 4. RgaPreprocessor olustur
    RgaPreprocessor rga;

    // Frame callback: decode edilen frame'i isle
    decoder.setFrameCallback([&](DmaBufferPtr decoded_frame) {
        if (!g_running.load()) return;
        if (!decoded_frame || !decoded_frame->virt_addr) return;

        int cnt = frame_count.fetch_add(1);

        // Format tespitini ilk frame'de yap
        if (cnt == 0) {
            original_width = decoded_frame->width;
            original_height = decoded_frame->height;

            // Decoder DmaBuffer zaten format alanını doldurdu:
            // buf->format = (base_fmt == MPP_FMT_YUV422SP) ? PixelFormat::NV16
            //                                              : PixelFormat::NV12;
            detected_format = decoded_frame->format;

            // Streameri gerçek frame boyutlarıyla güncelle
            streamer.setDimensions(original_width, original_height);

            std::cout << "[main] Orijinal boyutlar: "
                      << original_width << "x" << original_height << "\n";
            std::cout << "[main] Tespit edilen format: ";
            if (detected_format == PixelFormat::NV16) {
                std::cout << "NV16\n";
                std::cout << "[main] ====> RGA NV16->RGB888 + Stream NV16 kullanacak\n";
            } else {
                std::cout << "NV12\n";
                std::cout << "[main] ====> RGA NV12->RGB888 + Stream NV12 kullanacak\n";
            }
        }

        std::lock_guard<std::mutex> lock(g_frame_mutex);

        // ================================================================
        // ADIM 1: YUV (NV12 veya NV16) -> 640x640 RGB888 (RGA)
        // ================================================================
        size_t rgb640_size = static_cast<size_t>(target_w) * target_h * 3;
        DmaBufferPtr rgb640 = rga.acquireOutputBuffer(
            rgb640_size, target_w, target_h, PixelFormat::RGB888);
        if (!rgb640) {
            std::cerr << "[main] RGB640 DMA buffer tahsisi basarisiz.\n";
            return;
        }

        bool converted;
        if (detected_format == PixelFormat::NV16) {
            converted = rga.nv16ToRgb888(decoded_frame, original_width, original_height,
                                          rgb640, target_w, target_h);
        } else {
            converted = rga.nv12ToRgb888(decoded_frame, original_width, original_height,
                                          rgb640, target_w, target_h);
        }

        if (!converted) {
            std::cerr << "[main] YUV->RGB888 donusum basarisiz (fmt="
                      << (detected_format == PixelFormat::NV16 ? "NV16" : "NV12") << ").\n";
            return;
        }

        // ================================================================
        // ADIM 2: 640x640 RGB'yi PPM olarak kaydet (renkli dogrulama)
        // ================================================================
        if (cnt < 10 && cnt % 5 == 0) {
            std::string ppm_path = "/tmp/ph4_frame_rgb640_" + std::to_string(cnt) + ".ppm";
            RgaPreprocessor::saveRgb888ToPpm(rgb640, ppm_path, target_w, target_h);
        }

        // ================================================================
        // ADIM 3: RGB uzerine 3 adet dummy BBox ciz (RGA ile)
        // ================================================================
        bool bbox_drawn = rga.draw3DummyBboxes(rgb640, target_w, target_h);
        if (!bbox_drawn) {
            std::cerr << "[main] BBox cizimi basarisiz.\n";
        }

        // ================================================================
        // ADIM 4: BBox'li RGB'yi PPM olarak kaydet (dogrulama)
        // ================================================================
        if (cnt < 3) {
            std::string bbox_ppm_path = "/tmp/ph4_frame_bbox_rgb640_" + std::to_string(cnt) + ".ppm";
            RgaPreprocessor::saveRgb888ToPpm(rgb640, bbox_ppm_path, target_w, target_h);
        }

        // ================================================================
        // ADIM 5: RGB888 (640x640) -> YUV (orijinal boyut, tespit edilen formatla)
        // ================================================================
        // Rockchip RGA/MPP: width ve height 16'ya hizalanmalidir (chroma padding)
        auto align16 = [](uint32_t v) { return ((v + 15) / 16) * 16; };
        uint32_t aligned_w = align16(original_width);
        uint32_t aligned_h = align16(original_height);

        size_t yuv_orig_size;
        PixelFormat out_fmt = detected_format;
        if (out_fmt == PixelFormat::NV16) {
            // NV16: 2 byte/pixel (YUV 4:2:2)
            yuv_orig_size = static_cast<size_t>(aligned_w) * aligned_h * 2;
        } else {
            // NV12: 1.5 byte/pixel (YUV 4:2:0)
            yuv_orig_size = static_cast<size_t>(aligned_w) * aligned_h * 3 / 2;
        }

        DmaBufferPtr yuv_processed = rga.acquireOutputBuffer(
            yuv_orig_size, aligned_w, aligned_h, out_fmt);
        if (!yuv_processed) {
            std::cerr << "[main] Output YUV DMA buffer tahsisi basarisiz.\n";
            return;
        }

        bool back_converted;
        if (out_fmt == PixelFormat::NV16) {
            back_converted = rga.rgb888ToNv16(rgb640, target_w, target_h,
                                               yuv_processed, aligned_w, aligned_h);
        } else {
            back_converted = rga.rgb888ToNv12(rgb640, target_w, target_h,
                                               yuv_processed, aligned_w, aligned_h);
        }

        if (!back_converted) {
            std::cerr << "[main] RGB888->YUV geri donusum basarisiz.\n";
            return;
        }

        // ================================================================
        // ADIM 6: Islenmis YUV frame'i stream'e gonder (color.md §5a ile uyumlu)
        // mpph264enc her zaman NV12 bekler — pushFrame() otomatik NV16→NV12 downsample yapar
        // ================================================================
        streamer.pushFrame(yuv_processed);

        if ((cnt+1) % 100 == 0) {
            std::cout << "[main] " << (cnt+1) << " islenmis frame stream'e gonderildi.\n";
        }
    });

    // Demuxer callback: her yeni paket geldiğinde decoder'a gönder
    demuxer.setPacketCallback([&](EncodedPacket&& packet) {
        if (!g_running.load()) return;
        decoder.feedPacket(std::move(packet));
    });

    std::cout << "[main] Pipeline baslatildi. Cikmak icin Ctrl+C basin.\n";
    std::cout << "[main] Renkli yayin: rtsp://<firefly-ip>:8554/live\n";

    // Main loop
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto stats = decoder.getStats();
        if (stats.frames_decoded % 300 == 0 && stats.frames_decoded > 0) {
            std::cout << "[main] Decoder: " << stats.frames_decoded << " frame decoded, "
                      << stats.frames_dropped << " dropped\n";
        }
    }

    std::cout << "[main] Pipeline durduruluyor...\n";

    decoder.stop();
    demuxer.stop();
    streamer.stop();

    auto final_stats = decoder.getStats();
    std::cout << "[main] Toplam " << final_stats.frames_decoded << " frame cozuldu.\n";
    std::cout << "[main] Pipeline kapatildi.\n";

    return 0;
}