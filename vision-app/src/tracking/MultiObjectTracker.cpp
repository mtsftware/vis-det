#include "tracking/MultiObjectTracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

MultiObjectTracker::MultiObjectTracker() = default;

void MultiObjectTracker::setConfig(const Config& cfg) {
    cfg_ = cfg;
}

std::vector<TrackableObject> MultiObjectTracker::update(
    const std::vector<YoloDetection>& detections,
    int frame_index) {

    frame_counter_++;

    // === ADIM 1: Eski objeleri tahmin et (constant velocity model) ===
    for (auto& [id, obj] : objects_) {
        if (!obj.active) continue;

        if (obj.lost_count > 0) {
            obj.lost_count++;
            if (obj.lost_count > cfg_.max_lost_frames) {
                obj.active = false;
                lost_objects_++;
                continue;
            }
        }

        // Constant velocity tahmini: son hareketi devam ettir
        if (obj.last_seen_frame < frame_index - 1 && obj.lost_count == 0) {
            float dt = static_cast<float>(frame_index - obj.last_seen_frame);
            float vx = (obj.pred_x - obj.detection.x - obj.detection.width / 2.0f) / std::max(dt, 1.0f);
            float vy = (obj.pred_y - obj.detection.y - obj.detection.height / 2.0f) / std::max(dt, 1.0f);

            float new_cx = obj.detection.x + obj.detection.width / 2.0f + vx;
            float new_cy = obj.detection.y + obj.detection.height / 2.0f + vy;

            obj.pred_x = new_cx;
            obj.pred_y = new_cy;
        }
    }

    // === ADIM 2: IOU match — Yeni detection'lari eski track'lere ata ===
    std::vector<bool> matched_detection(detections.size(), false);
    std::vector<int> matched_track_ids;

    for (const auto& det : detections) {
        float best_iou = 0.0f;
        int best_id = -1;

        for (auto& [id, obj] : objects_) {
            if (!obj.active) continue;
            if (obj.lost_count > cfg_.max_lost_frames / 2) continue;  // çok kaybolan ile match etme
            // COCO çok-sınıf: farklı sınıflar (örn. "person" vs "car")
            // birbirinin track ID'sini çalmasın — sadece aynı sınıf eşleşir.
            if (obj.detection.class_id != det.class_id) continue;

            // IoU hesapla
            float x1 = std::max(obj.detection.x, det.x);
            float y1 = std::max(obj.detection.y, det.y);
            float x2 = std::min(obj.detection.x + obj.detection.width, det.x + det.width);
            float y2 = std::min(obj.detection.y + obj.detection.height, det.y + det.height);

            float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area_obj = obj.detection.width * obj.detection.height;
            float area_det = det.width * det.height;
            float union_area = area_obj + area_det - inter;

            if (union_area < 1e-6f) continue;

            float iou = inter / union_area;

            if (iou > best_iou) {
                best_iou = iou;
                best_id = id;
            }
        }

        // IOU eşigi üzerinde mi?
        if (best_iou >= cfg_.iou_thresh && best_id >= 0) {
            auto& obj = objects_[best_id];
            obj.detection = det;
            obj.detection.track_id = best_id;
            obj.lost_count = 0;
            obj.last_seen_frame = frame_index;
            obj.pred_x = det.x + det.width / 2.0f;
            obj.pred_y = det.y + det.height / 2.0f;
            matched_detection[&det - detections.data()] = true;
            matched_track_ids.push_back(best_id);
        }
    }

    // === ADIM 3: Eşleşmeyen detection'lara yeni ID ata ===
    for (size_t i = 0; i < detections.size(); ++i) {
        if (matched_detection[i]) continue;

        int new_id = cfg_.next_id++;
        total_tracked_++;

        TrackableObject newObj;
        newObj.id = new_id;
        newObj.detection = detections[i];
        newObj.detection.track_id = new_id;
        newObj.pred_x = detections[i].x + detections[i].width / 2.0f;
        newObj.pred_y = detections[i].y + detections[i].height / 2.0f;
        newObj.pred_w = detections[i].width;
        newObj.pred_h = detections[i].height;
        newObj.lost_count = 0;
        newObj.last_seen_frame = frame_index;
        newObj.active = true;

        objects_[new_id] = newObj;
    }

    // === ADIM 4: Sonuçları döndür ===
    std::vector<TrackableObject> result;
    result.reserve(objects_.size());

    for (auto& [id, obj] : objects_) {
        if (obj.active) {
            result.push_back(obj);
        }
    }

    return result;
}

std::vector<TrackableObject> MultiObjectTracker::getActiveObjects() const {
    std::vector<TrackableObject> result;
    for (const auto& [id, obj] : objects_) {
        if (obj.active) {
            result.push_back(obj);
        }
    }
    return result;
}

int MultiObjectTracker::activeCount() const {
    int count = 0;
    for (const auto& [id, obj] : objects_) {
        if (obj.active) count++;
    }
    return count;
}

int MultiObjectTracker::totalTracked() const {
    return total_tracked_;
}

int MultiObjectTracker::lostCount() const {
    return lost_objects_;
}