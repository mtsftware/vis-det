#pragma once

#include "ingestion/RtspDemuxer.hpp"  // EncodedPacket
#include "types/DmaBuffer.hpp"
#include "types/VideoCoding.hpp"

#include <cstdint>
#include <functional>
#include <memory>

// KRİTİK MİMARİ KURAL: mppvideodec (GStreamer elementi) DEĞİL, doğrudan
// rk_mpi.h üzerinden native MPP API'si kullanılıyor (06-system-architecture.md,
// Rapor 1 "Hibrit Mimari" kararı). Çıktı, MPP'nin kendi buffer group'undan
// (MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_DMA32) gelen zero-copy DmaBuffer'dır.
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
        uint64_t frames_dropped = 0;  // hasarlı/discard işaretli kareler
        uint64_t packets_failed = 0;  // decode_put_packet kalıcı hata
    };
    Stats getStats() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};
