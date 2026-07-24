#pragma once

#include "postprocess/YoloPostProcessor.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

// Multi-Object Tracker — CTCCTracker pattern (IOU match + state predict).
// Her detection'a benzersiz ID atar, kaybolan objeleri izler.
//
// Referans: tracking-app/src/tracking/LightTrackImpl.cpp (tek hedef tracking
// mantigindan esinlenerek, çok hedefe genisletildi).

struct TrackableObject {
    int id = -1;
    YoloDetection detection;       // son bilinen durum
    float pred_x = 0;              // predicted center-x
    float pred_y = 0;              // predicted center-y
    float pred_w = 0;              // predicted width
    float pred_h = 0;              // predicted height
    int lost_count = 0;            // kaç frame görülmüyor
    int last_seen_frame = 0;       // son Görülünen frame indexi
    bool active = true;            // hala aktif mi
};

class MultiObjectTracker {
public:
    MultiObjectTracker();
    ~MultiObjectTracker() = default;

    // Config ayarlari
    struct Config {
        float iou_thresh = 0.4f;      // match IOU esigi
        int max_lost_frames = 10;     // kaybolma toleransi (frame)
        int min_detection_score = 0;  // NMS sonras minimum skor (tracker icin dusuk)
        int next_id = 1;              // yeni ID baslangici
    };

    void setConfig(const Config& cfg);

    // Yeni detection'lari isle ve ID'leri ata.
    // Frame index: kare sirasi (sifirdan baslar, her track() cagrisinda artar).
    // Donus: tracked object'lerin liste (ID'si atanmis detection'lar).
    std::vector<TrackableObject> update(
        const std::vector<YoloDetection>& detections,
        int frame_index);

    // Aktif objelerin son durumu dondur (render icin kaybolanlar dahil).
    std::vector<TrackableObject> getActiveObjects() const;

    // Stats
    int activeCount() const;
    int totalTracked() const;
    int lostCount() const;

private:
    Config cfg_;
    std::map<int, TrackableObject> objects_;  // id → obje
    int total_tracked_ = 0;
    int lost_objects_ = 0;
    uint64_t frame_counter_ = 0;
};