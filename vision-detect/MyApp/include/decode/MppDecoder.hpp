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
    };
    Stats getStats() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};