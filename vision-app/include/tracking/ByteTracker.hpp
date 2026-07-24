#pragma once

#include "postprocess/YoloPostProcessor.hpp"  // YoloDetection

#include <cstdint>
#include <memory>
#include <vector>

// ByteTrack — referans: edge-ai-workshop-rknn/bytetracker_fix.py (BYTETracker,
// STrack, KalmanFilter), kendisi ifzhang/ByteTrack (ECCV 2022) C++
// orijinalinin (BYTETracker.cpp/STrack.cpp/kalmanFilter.cpp) Python portu —
// burada da AYNI algoritma, C++'a geri cevrilmis. scipy.linear_sum_assignment
// yerine kendi Macar algoritmasi (e-maxx O(n^2*m) potansiyel yontemi)
// implementasyonu kullanilir (harici bagimlilik yok — CLAUDE.md "paket
// kurulumu yok" kurali).
//
// TEK ONEMLI DAVRANIS FARKI (kullanicinin istegi): kayip (Lost) track'ler
// referansta oldugu gibi ANINDA ciktidan (ve dolayisiyla cizimden) DUSER —
// eski MultiObjectTracker'daki gibi birkaç kare "kirmizi/soluk" gösterip
// beklemez. Kayip track kimligi yeniden-tanima icin dahili olarak
// max_time_lost kadar bellekte tutulur ama SADECE "Tracked" durumundaki
// track'ler update()'in dondurdugu listede yer alir (bkz. .cpp'deki
// "SADECE Tracked+is_activated" notu).

namespace ByteTrackConfig {
// edge-ai-workshop-rknn/config.json'daki "tracking" bolumuyle BIREBIR AYNI
// degerler (main.py: track_cfg.get(...) varsayilanlariyla da tutarli).
constexpr int kFrameRate = 30;     // video kare hizi
constexpr int kTrackBuffer = 30;   // kayip track'in bellekte tutulma suresi (kare, frame_rate'e olceklenir)
constexpr float kTrackThresh = 0.3f;  // yuksek/dusuk guven ayrimi esigi
constexpr float kHighThresh = 0.5f;   // yeni track baslatma esigi (sadece yuksek guvenli det'ler yeni ID alir)
constexpr float kMatchThresh = 0.8f;  // 1. asama (yuksek guven) IoU eslestirme esigi — cost=1-IoU <= bu

// Referansta sabit (config'e acilmamis, BYTETracker.update() icinde literal):
constexpr float kLowMatchThresh = 0.5f;          // 2. asama (dusuk guven) eslestirme esigi
constexpr float kUnconfirmedMatchThresh = 0.7f;  // onaylanmamis (henuz aktive olmamis) track esigi
constexpr float kDuplicateIouThresh = 0.15f;     // remove_duplicate_stracks esigi (dist < bu -> duplicate)
}  // namespace ByteTrackConfig

// STrack yasam dongusu durumu — referans: bytetracker_fix.py TrackState
// (IntEnum). ByteTracker.cpp'deki STrack bunu kullanir; ByteTracker::update()
// disina hic sizmaz (TrackedBox'ta yer almaz) ama .cpp'nin state'i
// kullanabilmesi icin burada (public header'da) tanimli olmasi gerekiyor.
enum class TrackState { New, Tracked, Lost, Removed };

struct TrackedBox {
    int track_id = -1;
    float x = 0;       // top-left x (Kalman-tahminli, orijinal frame piksel uzayi)
    float y = 0;       // top-left y
    float width = 0;
    float height = 0;
    float confidence = 0.0f;
    int class_id = 0;
};

class ByteTracker {
public:
    ByteTracker();
    ~ByteTracker();

    ByteTracker(const ByteTracker&) = delete;
    ByteTracker& operator=(const ByteTracker&) = delete;

    // detections: bu karenin YOLO tespitleri (letterbox ters-map edilmis,
    // orijinal frame piksel uzayinda — YoloPostProcessor::decodeOutputs +
    // applyNMS ciktisi dogrudan verilebilir).
    //
    // Donus: SADECE su an "Tracked" (aktif eslesmis/yeniden-bulunmus) ve
    // is_activated olan track'ler — kayip olan bir SONRAKI update()
    // cagrisinda ANINDA bu listeden duser (bekletme YOK).
    std::vector<TrackedBox> update(const std::vector<YoloDetection>& detections);

    // Istatistikler (PipelineOrchestrator::Stats ile uyumlu isimler).
    int activeCount() const;    // su anki ciktida donen track sayisi
    int totalTracked() const;   // simdiye kadar olusturulan toplam benzersiz track ID sayisi
    int lostCount() const;      // kalici olarak silinmis (Removed) track sayisi

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};
