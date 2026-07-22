#pragma once

#include "tracking/ITracker.hpp"
#include "types/DmaBuffer.hpp"
#include "types/VideoCoding.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// 06-system-architecture.md §5'in thread modelinin somutlaşması. M6/M7
// fazlarında canlıda kanıtlanan desenin (CLAUDE.md "Canlıda Doğrulanmış
// Kurallar" §5) modül hali:
//
//   Ingestion (GStreamer thread'leri) -> MppDecoder (kendi put/get
//   thread'leri) -> [en-yeni-kare slotu, decode ASLA bloklanmaz, bayat
//   kare düşürülür] -> TEK seri tracking thread (ITracker::track,
//   05-ITracker.md §9: ardışık kareler paralelleştirilemez).
//
// Sorumluluk sınırı:
//   Bilir:  demuxer/decoder yaşam döngüsünü, kare akışını, tracker'ın
//           initialize/track çağrı disiplinini, LOST sinyalini.
//   Bilmez: çizim/yayın/encode (çıktı katmanı kapsam dışı — 06 §9.4;
//           sonuçlar ResultCallback ile dışarı verilir), RGA/RKNN iç
//           detayları (ITracker arkasında), detector'ın kendisi (sadece
//           LOST sinyalini yayınlar — detector geldiğinde bu sinyalin
//           tüketicisi olacak, bkz. 06 §5 Detector Thread).
struct PipelineConfig {
    std::string rtsp_url;
    uint32_t rtsp_latency_ms = 100;
    VideoCoding coding = VideoCoding::H264;
};

struct FrameResult {
    DmaBufferPtr frame;      // decode çıktısı (çizim/yayın için)
    TrackResult track;       // tracked=true ise geçerli
    bool tracked = false;    // false: henüz initialize edilmedi (ham kare)
    TrackerState state = TrackerState::Idle;
};

class PipelineOrchestrator {
public:
    // TRACKING THREAD'İNDE çağrılır — hafif tutulmalı (çizim + kuyruğa
    // bırakma gibi). Ağır iş burada koşarsa tracking FPS'i ona kilitlenir
    // (decode ETKİLENMEZ — slot deseni decode'u zaten izole ediyor).
    using ResultCallback = std::function<void(const FrameResult&)>;

    // Hedef kaybolduğunda her LOST karesinde çağrılır (ardışık LOST kare
    // sayısıyla). Detector entegrasyonunun bağlanacağı kanca: tüketici
    // detector'ı tetikleyip bulduğu bbox ile requestInitialize() çağırır
    // (05-ITracker.md §6'daki seçim mantığı TÜKETİCİNİN işi, bizim değil).
    using LostCallback = std::function<void(uint32_t consecutive_lost)>;

    // tracker: modelleri YÜKLENMİŞ olmalı (LightTrackImpl::loadModels
    // çağıran tarafın işi — model yüklemeden önce ana thread'i
    // pinCurrentThreadToBigCores() ile pinlemeyi unutma, CLAUDE.md kural #6).
    PipelineOrchestrator(ITracker& tracker, PipelineConfig config);
    ~PipelineOrchestrator();

    PipelineOrchestrator(const PipelineOrchestrator&) = delete;
    PipelineOrchestrator& operator=(const PipelineOrchestrator&) = delete;

    void setResultCallback(ResultCallback cb);
    void setLostCallback(LostCallback cb);

    // Demuxer+decoder'ı kurar/başlatır, tracking thread'ini açar.
    bool start();
    void stop();

    // İlk hedef (CLI'dan) ya da yeniden-edinim (ileride detector'dan).
    // Thread-safe; initialize() bir SONRAKİ karede, tracking thread'inde
    // koşar (ITracker'ın tek-thread kuralı korunur — initialize ve track
    // asla farklı thread'lerden çağrılmaz).
    void requestInitialize(const BBox& bbox);

    struct Stats {
        uint64_t frames_decoded = 0;
        uint64_t frames_tracked = 0;
        uint64_t frames_dropped = 0;   // slot doluyken gelen (bayat) kareler
        uint64_t frames_lost = 0;      // LOST durumundaki track() sonuçları
        float last_confidence = 0.0f;
    };
    Stats getStats() const;

    // CLAUDE.md kural #6 yardımcıları: RKNN iç işçi thread'leri rknn_init
    // anında affinity miras alır — model yüklemesi bu ikisinin arasında
    // yapılmalı. (Orkestratörün kendi tracking thread'i kendini zaten pinler.)
    static void pinCurrentThreadToBigCores();
    static void restoreCurrentThreadAllCores();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};