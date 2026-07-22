#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Sıkıştırılmış (H.264 NAL) tek bir paket. Bu aşamada ham piksel yok,
// dolayısıyla zero-copy/DMA-BUF kısıtları henüz devrede değil — küçük
// sıkıştırılmış veri için CPU kopyası kabul edilebilir.
struct EncodedPacket {
    std::vector<uint8_t> data;
    uint64_t pts_ns = 0;      // GStreamer PTS (geçerliyse), yoksa 0
    uint64_t arrival_ns = 0;  // Bu makinede alındığı an (monotonic)
};

// KRİTİK MİMARİ KURAL: Bu sınıf SADECE ağ/RTP/demux katmanını yönetir.
// mppvideodec veya rgaconvert kullanmaz — decode ve scale işlemleri
// bilerek native MPP/RGA API'lerine bırakılmıştır. appsink'ten çıkan veri,
// çözülmemiş H.264 Annex-B akışıdır.
class RtspDemuxer {
public:
    using PacketCallback = std::function<void(EncodedPacket&&)>;

    // latency_ms: rtspsrc jitter buffer büyüklüğü. Düşük değer = düşük
    // gecikme ama ağ sağlıksızsa (WiFi, paket sırası bozulması) risklidir.
    explicit RtspDemuxer(std::string rtsp_url, uint32_t latency_ms = 100);
    ~RtspDemuxer();

    RtspDemuxer(const RtspDemuxer&) = delete;
    RtspDemuxer& operator=(const RtspDemuxer&) = delete;

    // Her yeni paket geldiğinde çağrılır. start() öncesi set edilmelidir.
    void setPacketCallback(PacketCallback cb);

    bool start();
    void stop();

    struct Stats {
        uint64_t packets_received = 0;
        uint64_t bytes_received = 0;
        uint64_t packets_dropped = 0;  // appsink buffer map hatası
    };
    Stats getStats() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};