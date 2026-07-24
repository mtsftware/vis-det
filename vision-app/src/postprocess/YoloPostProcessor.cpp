#include "postprocess/YoloPostProcessor.hpp"

#include "tracking/MultiObjectTracker.hpp"

#include "im2d.h"
#include "rga.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

YoloPostProcessor::YoloPostProcessor() = default;

// Model output [1, 5, 8400] CHANNEL-FIRST (NCHW) uyumlu decode:
//   Channel 0: cx, Channel 1: cy, Channel 2: w, Channel 3: h, Channel 4: score
//   Bellek layoutu: data[channel * 8400 + anchor_index]
//   Skor kanali ONNX grafiginde zaten Sigmoid'den gecmis geliyor
//   (verification/read.md) — burada ELLE sigmoid UYGULANMAZ.
bool YoloPostProcessor::decodeOutputs(const void* raw_output, uint32_t raw_size,
                                       int orig_w, int orig_h,
                                       const LetterboxResult& letterbox,
                                       float conf_thresh,
                                       std::vector<YoloDetection>& detections) {
    if (!raw_output || raw_size == 0) {
        std::cerr << "[YoloPostProcessor] Bos cikti\n";
        return false;
    }
    if (letterbox.ratio < 1e-9) {
        std::cerr << "[YoloPostProcessor] Gecersiz letterbox ratio\n";
        return false;
    }

    const float* data = static_cast<const float*>(raw_output);

    // Model output: [1, 5, 8400]
    // CH (channel count): 5 = 4 (box) + 1 (score)
    // NA (num anchors): 8400 (640x640 girdi icin 80x80 + 40x40 + 20x20)
    const int NA = 8400;

    const double inv_ratio = 1.0 / letterbox.ratio;

    detections.clear();
    detections.reserve(256);

    for (int i = 0; i < NA; ++i) {
        float score = data[4 * NA + i];
        if (score < conf_thresh) continue;

        // Box degerlerini al (center-x, center-y, width, height) — 640x640
        // letterbox uzayinda.
        float cx = data[0 * NA + i];
        float cy = data[1 * NA + i];
        float w  = data[2 * NA + i];
        float h  = data[3 * NA + i];

        // center-x,y -> top-left x,y (hala letterbox uzayinda)
        float lx = cx - w / 2.0f;
        float ly = cy - h / 2.0f;

        // Letterbox TERS donusum: pad offset'ini cikar, ratio'ya bol —
        // duz stretch/scale_x-scale_y YERINE (RgaPreprocessor::process()
        // ile birebir tutarli tersleme).
        float x = static_cast<float>((lx - letterbox.dst_offset_x) * inv_ratio);
        float y = static_cast<float>((ly - letterbox.dst_offset_y) * inv_ratio);
        float bw = static_cast<float>(w * inv_ratio);
        float bh = static_cast<float>(h * inv_ratio);

        // Frame sinirlari icinde tut
        x = std::max(0.0f, x);
        y = std::max(0.0f, y);
        bw = std::min(bw, static_cast<float>(orig_w) - x);
        bh = std::min(bh, static_cast<float>(orig_h) - y);

        // Minimum boyut kontrolü
        if (bw < 1.0f || bh < 1.0f) continue;

        YoloDetection det;
        det.x = x;
        det.y = y;
        det.width = bw;
        det.height = bh;
        det.confidence = score;
        det.class_id = 0;  // tek-class model
        det.track_id = -1;

        detections.push_back(det);
    }

    return !detections.empty();
}

std::vector<YoloDetection> YoloPostProcessor::applyNMS(std::vector<YoloDetection>& detections,
                                                          float iou_thresh) {
    if (detections.empty()) return {};

    // confidence'e göre sırala
    std::sort(detections.begin(), detections.end(),
              [](const YoloDetection& a, const YoloDetection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<YoloDetection> result;
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            // DIoU (plain IoU DEĞİL) — merkez mesafesi de cezalandırılır.
            float diou = calculateDiou(detections[i], detections[j]);
            if (diou > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

namespace {

int toRgaFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::NV12:
            return RK_FORMAT_YCbCr_420_SP;
        case PixelFormat::NV16:
            return RK_FORMAT_YCbCr_422_SP;
        case PixelFormat::RGB888:
            return RK_FORMAT_RGB_888;
        case PixelFormat::BGR888:
            return RK_FORMAT_BGR_888;
        default:
            return RK_FORMAT_YCbCr_420_SP;
    }
}

// Referans: tracking-app main_m11_test.cpp colorForConfidence() ile ayni
// esik/renk semasi (kirmizi/turuncu/yesil), artik takip durumuna gore.
int colorForTrack(float confidence, bool lost) {
    constexpr float kGreenThresh = 0.7f;
    constexpr float kOrangeThresh = 0.4f;
    if (lost || confidence <= kOrangeThresh) {
        return (255 << 16) | (0 << 8) | 0;  // kirmizi — kaybolan/dusuk guven
    }
    if (confidence <= kGreenThresh) {
        return (255 << 16) | (165 << 8) | 0;  // turuncu — orta guven
    }
    return (0 << 16) | (255 << 8) | 0;  // yesil — yuksek guven
}

im_rect alignRectEven(im_rect r) {
    r.x -= (r.x & 1);
    r.y -= (r.y & 1);
    r.width -= (r.width & 1);
    r.height -= (r.height & 1);
    if (r.width < 2) r.width = 2;
    if (r.height < 2) r.height = 2;
    return r;
}

}  // namespace

// Referans: tracking-app main_m11_test.cpp drawBboxOutline() ile AYNI teknik
// (RGA imfillArray, 4 ince kenar rect'i, 2-hizali). Referans tek hedef
// ciziyordu; burada MOT icin TUM aktif track'ler donguyle cizilir.
void YoloPostProcessor::drawTrackedObjects(const DmaBufferPtr& frame,
                                            const std::vector<TrackableObject>& objects,
                                            int line_thickness) {
    if (!frame || frame->fd < 0 || objects.empty()) return;

    int fw = static_cast<int>(frame->width);
    int fh = static_cast<int>(frame->height);

    rga_buffer_t buf = wrapbuffer_fd(frame->fd, fw, fh, toRgaFormat(frame->format),
                                      static_cast<int>(frame->w_stride),
                                      static_cast<int>(frame->h_stride));

    static bool logged_error = false;

    for (const auto& obj : objects) {
        if (!obj.active) continue;

        int x0 = std::max(0, static_cast<int>(obj.detection.x));
        int y0 = std::max(0, static_cast<int>(obj.detection.y));
        int x1 = std::min(fw, static_cast<int>(obj.detection.x + obj.detection.width));
        int y1 = std::min(fh, static_cast<int>(obj.detection.y + obj.detection.height));
        if (x1 <= x0 || y1 <= y0) continue;

        int t = std::min(line_thickness, std::min(x1 - x0, y1 - y0) / 2);
        if (t < 1) t = 1;

        bool lost = obj.lost_count > 0;
        int color = colorForTrack(obj.detection.confidence, lost);

        im_rect rects[4] = {
            alignRectEven(im_rect{x0, y0, x1 - x0, t}),
            alignRectEven(im_rect{x0, std::max(y0, y1 - t), x1 - x0, t}),
            alignRectEven(im_rect{x0, y0, t, y1 - y0}),
            alignRectEven(im_rect{std::max(x0, x1 - t), y0, t, y1 - y0}),
        };

        IM_STATUS r = imfillArray(buf, rects, 4, static_cast<uint32_t>(color));
        if (!logged_error && r != IM_STATUS_SUCCESS) {
            std::cerr << "[YoloPostProcessor] drawTrackedObjects: imfillArray başarısız: "
                      << imStrError(r) << " (bir kez loglanıyor)\n";
            logged_error = true;
        }
    }
}
