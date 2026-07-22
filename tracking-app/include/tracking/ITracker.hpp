#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>

// 05-ITracker.md sözleşmesinin soyut arayüzü. LightTrackImpl, ileride
// NanoTrackImpl bu arayüzü karşılayacak somut sınıflardır. Bu dosya
// hiçbir tracker'a özgü sabit/formül BİLMEZ — hepsi somut implementasyonda
// kapalı kalır (kullanıcı isteği: yeni bir tracker eklemek hızlı olsun).
struct BBox {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct TrackResult {
    BBox bbox;
    float confidence = 0.0f;
};

enum class TrackerState { Idle, Tracking, Lost };

// Son track() çağrısının aşama-bazlı süre dökümü (mikrosaniye) — mLogger_test.cpp
// (perf_logger_v2.py'nin C++ karşılığı) tarafından invoke/postproc metriklerini
// doldurmak için okunur. Tracker-özel değil (her iki somut tracker da AYNI
// aşamalara sahip: RGA crop -> NPU backbone+head -> CPU decode), bu yüzden
// ITracker'da genel bir sözleşme olarak duruyor.
struct TrackTiming {
    uint64_t crop_us = 0;      // RGA arama-alanı crop+resize (girdi hazırlama)
    uint64_t invoke_us = 0;    // backbone + head NPU çalışma süresi toplamı
    uint64_t postproc_us = 0;  // sigmoid/penalty/argmax/bbox decode (CPU)
};

class ITracker {
public:
    virtual ~ITracker() = default;

    // frame: tam çözünürlüklü kaynak (kırpma işlemi için RGA'nın okuyacağı
    // DmaBuffer — decode çıktısı, henüz hiçbir dönüşüme uğramamış).
    // bbox: ilk hedef konumu (detector'dan ya da elle verilmiş).
    virtual bool initialize(const DmaBufferPtr& frame, const BBox& bbox) = 0;

    // Her karede çağrılır — TEK, SERİ thread üzerinden (05-ITracker.md §9:
    // durumlu modeller ardışık-kare paralelleştirmesine tabi tutulamaz).
    // Durum LOST'a düşerse (confidence < eşik), dönen sonucun confidence'ı
    // düşük olur ama fonksiyon "başarısız" anlamına gelmez — durumu
    // getState() ile ayrıca sorgula.
    virtual TrackResult track(const DmaBufferPtr& frame) = 0;

    virtual TrackerState getState() const = 0;

    // Son BAŞARILI track() çağrısının zamanlama dökümü — bkz. TrackTiming.
    // track() hiç çağrılmadıysa ya da bir aşama başarısız olup erken
    // dönmüşse, o aşamanın değeri bir ÖNCEKİ başarılı ölçümde kalır (sıfır
    // sıçraması yerine son bilinen değer — grafikte tek atlanan kareyi
    // yanıltıcı bir çukur olarak göstermemek için).
    virtual TrackTiming lastTiming() const = 0;

    virtual void shutdown() = 0;
};