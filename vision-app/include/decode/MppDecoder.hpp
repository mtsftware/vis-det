#pragma once

#include "ingestion/RtspDemuxer.hpp"  // EncodedPacket
#include "types/DmaBuffer.hpp"
#include "types/VideoCoding.hpp"

#include <cstdint>
#include <functional>
#include <memory>

// MppDecoder: H.264/H.265 NAL paketlerini NV12 YUV frame'lere çeviren
// Rockchip MPP (Media Processing Platform) decoder sarmalayıcısı.
// Native MPP API kullanır (GStreamer mppvideodec yerine).
// Çıktı: DmaBufferPtr ile zero-copy NV12 frame callback.
class MppDecoder {
public:
    using FrameCallback = std::function<void(DmaBufferPtr)>;

    explicit MppDecoder(VideoCoding coding = VideoCoding::H264);
    ~MppDecoder();

    MppDecoder(const MppDecoder&) = delete;
    MppDecoder& operator=(const MppDecoder&) = delete;

    void setFrameCallback(FrameCallback cb);
    bool start();
    void stop();

    // RtspDemuxer::PacketCallback ile doğrudan uyumlu.
    void feedPacket(EncodedPacket&& packet);

    struct Stats {
        uint64_t frames_decoded = 0;
        uint64_t frames_dropped = 0;
        uint64_t packets_failed = 0;
        // Son basarili mpi->decode_get_frame() cagrisinin surdugu sure —
        // NOT: MPP put/get thread'leri ayri (asenkron) oldugu icin bu, NPU
        // benzeri "saf decode suresi" degil, sadece kareyi COZULMUS olarak
        // ALMA cagrisinin gecikmesidir (havuzda bekleme dahil olabilir).
        // Yine de pratikte decode darbogazi teshisi icin kullanisli bir
        // proxy metrik.
        double last_decode_ms = 0.0;
    };
    Stats getStats() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};