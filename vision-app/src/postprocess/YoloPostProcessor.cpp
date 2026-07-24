#include "postprocess/YoloPostProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

YoloPostProcessor::YoloPostProcessor() = default;

// Model output [1, 5, 8400] CHANNE-FIRST (NCHW) uyumlu decode:
//   Channel 0: cx, Channel 1: cy, Channel 2: w, Channel 3: h, Channel 4: score
//   Bellek layoutu: data[channel * 8400 + anchor_index]
bool YoloPostProcessor::decodeOutputs(const void* raw_output, uint32_t raw_size,
                                       int orig_w, int orig_h,
                                       int model_w, int model_h,
                                       float conf_thresh,
                                       std::vector<YoloDetection>& detections) {
    if (!raw_output || raw_size == 0) {
        std::cerr << "[YoloPostProcessor] Bos cikti\n";
        return false;
    }

    const float* data = static_cast<const float*>(raw_output);

    // Model output: [1, 5, 8400]
    // CH (channel count): 5 = 4 (box) + 1 (score)
    // NA (num anchors): 8400
    const int NA = 8400;

    // Orijinal frame'e mapping icin scale faktörleri
    float scale_x = static_cast<float>(orig_w) / model_w;
    float scale_y = static_cast<float>(orig_h) / model_h;

    detections.clear();
    detections.reserve(256);  // tahmini rezervasyon
    
    float global_max_score = -1.0f;  // En yüksek skor (negative ile basla, her durumda log yaz)
    
    // Debug: ilk 5 score degerini yazdir (channel 4 = data[4*8400 + i])
    std::cout << "[YoloPostProcessor] Ilk 5 score degeri: ";
    for (int i = 0; i < 5; ++i) {
        float s = data[4 * NA + i];
        std::cout << s << " ";
    }
    std::cout << "\n";

    for (int i = 0; i < NA; ++i) {
        // Box degerlerini al (center-x, center-y, width, height)
        float cx = data[0 * NA + i];
        float cy = data[1 * NA + i];
        float w  = data[2 * NA + i];
        float h  = data[3 * NA + i];
        float score = data[4 * NA + i];

        // Global maximum skoru takip et (threshold filtrelemesi ÖNCE)
        if (score > global_max_score) {
            global_max_score = score;
        }

        // Confidence threshold altindaki anchor'lari atla
        if (score < conf_thresh) continue;

        // Bbox format dönüþümü: center-x,y → top-left x,y
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        // Orijinal frame boyutuna map et
        x *= scale_x;
        y *= scale_y;
        w *= scale_x;
        h *= scale_y;

        // Frame sinirlari icinde tut
        x = std::max(0.0f, x);
        y = std::max(0.0f, y);
        w = std::min(w, static_cast<float>(orig_w) - x);
        h = std::min(h, static_cast<float>(orig_h) - y);

        // Minimum boyut kontrolü
        if (w < 1.0f || h < 1.0f) continue;

        YoloDetection det;
        det.x = x;
        det.y = y;
        det.width = w;
        det.height = h;
        det.confidence = score;
        det.class_id = 0;  // tek-class model
        det.track_id = -1;

        detections.push_back(det);
    }

    // Maksimum skor logu (her durumda yazdir — global_max_score baslangicta -1.0f)
    if (global_max_score >= 0.0f) {
        std::cout << "[YoloPostProcessor] Frame - Bulunan en yüksek skor (Max Confidence): "
                  << global_max_score << "\n";
    } else {
        std::cout << "[YoloPostProcessor] Frame - hicbir skor 0.0f'u asmedi (max: -1.0f)\n";
    }
    
    return !detections.empty();
}

std::vector<YoloDetection> YoloPostProcessor::applyNMS(std::vector<YoloDetection>& detections,
                                                          float iou_thresh) {
    if (detections.empty()) return {};

    //confidences'e göre sýrala
    std::sort(detections.begin(), detections.end(),
              [](const YoloDetection& a, const YoloDetection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<YoloDetection> result;
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        // Bu bbox priority ile sonuca ekle
        result.push_back(detections[i]);

        // Sonraki bbox'larla IoU hesapla ve supprese et
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            float x1 = std::max(detections[i].x, detections[j].x);
            float y1 = std::max(detections[i].y, detections[j].y);
            float x2 = std::min(detections[i].x + detections[i].width,
                               detections[j].x + detections[j].width);
            float y2 = std::min(detections[i].y + detections[i].height,
                               detections[j].y + detections[j].height);

            float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area_i = detections[i].width * detections[i].height;
            float area_j = detections[j].width * detections[j].height;
            float union_area = area_i + area_j - inter;

            if (union_area < 1e-6f) continue;

            float iou = inter / union_area;
            if (iou > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}