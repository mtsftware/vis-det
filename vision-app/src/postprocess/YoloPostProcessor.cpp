#include "postprocess/YoloPostProcessor.hpp"

#include "tracking/ByteTracker.hpp"

#include "im2d.h"
#include "rga.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

YoloPostProcessor::YoloPostProcessor() = default;

namespace {
constexpr int kNumScales = 3;
constexpr int kRegMax = 16;
constexpr int kStrides[kNumScales] = {8, 16, 32};
}  // namespace

// airockchip RKNN-optimised YOLOv8 export decode — referans:
// edge-ai-workshop-rknn/inference.py (_postprocess_rknn + _dfl_decode).
// 9 cikti: [box_s,score_s,sum_s] x 3 olcek (stride 8/16/32). box_s [1,64,H,W]
// NCHW = 4 kenar x REG_MAX=16 DFL bin (softmax + agirlikli beklenti ile
// piksel mesafesine cevrilir); score_s [1,nc,H,W] NCHW zaten Sigmoid'den
// gecmis (burada TEKRAR sigmoid UYGULANMAZ).
bool YoloPostProcessor::decodeOutputs(const YoloInferenceEngine& engine,
                                       int orig_w, int orig_h,
                                       const LetterboxResult& letterbox,
                                       float conf_thresh,
                                       std::vector<YoloDetection>& detections,
                                       float* out_max_score) {
    detections.clear();
    detections.reserve(256);
    float max_score = -1.0f;

    if (engine.outputCount() < kNumScales * 3) {
        std::cerr << "[YoloPostProcessor] beklenen >=" << (kNumScales * 3)
                  << " RKNN cikti tensoru, alinan " << engine.outputCount() << "\n";
        return false;
    }
    if (letterbox.ratio < 1e-9) {
        std::cerr << "[YoloPostProcessor] Gecersiz letterbox ratio\n";
        return false;
    }

    const double inv_ratio = 1.0 / letterbox.ratio;
    float dfl_bins[kRegMax];
    // Olcekler arasi yeniden kullanilir (assign() ile boyutlandirilir) —
    // her karede/olcekte yeniden tahsis etmemek icin donguden disarida.
    static thread_local std::vector<float> best_score_buf;
    static thread_local std::vector<int> best_cls_buf;

    for (int s = 0; s < kNumScales; ++s) {
        const void* box_ptr = nullptr;
        uint32_t box_size = 0;
        const void* score_ptr = nullptr;
        uint32_t score_size = 0;
        // outputs[s*3+2] (sum) kullanilmiyor, atlanir.
        if (!engine.getOutput(s * 3 + 0, box_ptr, box_size)) continue;
        if (!engine.getOutput(s * 3 + 1, score_ptr, score_size)) continue;
        if (!box_ptr || !score_ptr) continue;

        const auto& score_info = engine.outputInfo(s * 3 + 1);
        // NCHW dims: [1, nc, H, W]
        if (score_info.n_dims < 4) {
            std::cerr << "[YoloPostProcessor] scale " << s << ": score tensor n_dims<4\n";
            continue;
        }
        int nc = score_info.dims[1];
        int H = score_info.dims[2];
        int W = score_info.dims[3];
        if (nc <= 0 || H <= 0 || W <= 0) continue;

        const int HW = H * W;
        const int stride = kStrides[s];
        const float* box = static_cast<const float*>(box_ptr);      // [64, H, W]
        const float* score = static_cast<const float*>(score_ptr);  // [nc, H, W]

        // PERFORMANS: sinif-max hesabini SINIF-MAJOR (c disarida, a icinde)
        // sirayla yap — score[c*HW+a] boylece c sabitken a ARDIL (contiguous)
        // okunur. Onceki surum a-disarida/c-iceride donuyordu; score[c*HW+a]
        // her c adiminda HW (400-6400) eleman ATLAYARAK okunuyordu — L1/L2
        // once bellek acisindan cok kotu bir erisim deseniydi ve
        // postprocess'in olcumlerde ~10ms'ye cikmasinin ana sebebiydi
        // (8400 anchor x 80 sinif = 672.000 tamamen sicramali okuma).
        // Simdi HER SINIF ICIN HW'lik ardil bir satir taranir (kalibre
        // edilmis onbellek dostu erisim), toplam is ayni ama bellek deseni
        // COK daha hizli.
        best_score_buf.assign(static_cast<size_t>(HW), -std::numeric_limits<float>::infinity());
        best_cls_buf.assign(static_cast<size_t>(HW), -1);
        for (int c = 0; c < nc; ++c) {
            const float* score_c = score + static_cast<size_t>(c) * HW;  // ardil HW eleman
            for (int a = 0; a < HW; ++a) {
                float sc = score_c[a];
                if (sc > best_score_buf[static_cast<size_t>(a)]) {
                    best_score_buf[static_cast<size_t>(a)] = sc;
                    best_cls_buf[static_cast<size_t>(a)] = c;
                }
            }
        }

        for (int a = 0; a < HW; ++a) {
            float best_score = best_score_buf[static_cast<size_t>(a)];
            int best_cls = best_cls_buf[static_cast<size_t>(a)];
            if (best_score > max_score) max_score = best_score;
            if (best_score < conf_thresh) continue;

            // DFL decode: 4 kenar (left,top,right,bottom), her biri REG_MAX
            // bin — bellek layoutu [64,H,W] NCHW oldugu icin bin'ler arasi
            // stride HW eleman (bitisik DEGIL).
            float ltrb[4];
            for (int side = 0; side < 4; ++side) {
                float bmax = -std::numeric_limits<float>::infinity();
                for (int b = 0; b < kRegMax; ++b) {
                    float v = box[(side * kRegMax + b) * HW + a];
                    dfl_bins[b] = v;
                    if (v > bmax) bmax = v;
                }
                float sum = 0.0f;
                for (int b = 0; b < kRegMax; ++b) {
                    float e = std::exp(dfl_bins[b] - bmax);
                    dfl_bins[b] = e;
                    sum += e;
                }
                float expectation = 0.0f;
                for (int b = 0; b < kRegMax; ++b) {
                    expectation += (dfl_bins[b] / sum) * static_cast<float>(b);
                }
                ltrb[side] = expectation * static_cast<float>(stride);
            }

            // Anchor merkezi (letterbox-uzayi piksel)
            int hh = a / W;
            int ww = a % W;
            float gx = (static_cast<float>(ww) + 0.5f) * stride;
            float gy = (static_cast<float>(hh) + 0.5f) * stride;

            float lx1 = gx - ltrb[0];
            float ly1 = gy - ltrb[1];
            float lx2 = gx + ltrb[2];
            float ly2 = gy + ltrb[3];

            // Letterbox TERS donusum: pad offset'ini cikar, ratio'ya bol.
            float x1 = static_cast<float>((lx1 - letterbox.dst_offset_x) * inv_ratio);
            float y1 = static_cast<float>((ly1 - letterbox.dst_offset_y) * inv_ratio);
            float x2 = static_cast<float>((lx2 - letterbox.dst_offset_x) * inv_ratio);
            float y2 = static_cast<float>((ly2 - letterbox.dst_offset_y) * inv_ratio);

            x1 = std::max(0.0f, x1);
            y1 = std::max(0.0f, y1);
            x2 = std::min(x2, static_cast<float>(orig_w));
            y2 = std::min(y2, static_cast<float>(orig_h));

            float bw = x2 - x1;
            float bh = y2 - y1;
            if (bw < 1.0f || bh < 1.0f) continue;

            YoloDetection det;
            det.x = x1;
            det.y = y1;
            det.width = bw;
            det.height = bh;
            det.confidence = best_score;
            det.class_id = best_cls;
            det.track_id = -1;

            detections.push_back(det);
        }
    }

    if (out_max_score) *out_max_score = max_score;

    return !detections.empty();
}

std::vector<YoloDetection> YoloPostProcessor::applyNMS(std::vector<YoloDetection>& detections,
                                                          float iou_thresh) {
    if (detections.empty()) return {};

    // Sinif ID'sine gore grupla (once sinif, sonra confidence azalan) — HER
    // SINIF ICINDE AYRI NMS uygulanir, farkli siniflarin ust uste binen
    // kutulari birbirini bastirmaz (referans: inference.py per-class NMS).
    std::sort(detections.begin(), detections.end(),
              [](const YoloDetection& a, const YoloDetection& b) {
                  if (a.class_id != b.class_id) return a.class_id < b.class_id;
                  return a.confidence > b.confidence;
              });

    std::vector<YoloDetection> result;
    result.reserve(detections.size());

    size_t i = 0;
    while (i < detections.size()) {
        size_t j = i;
        while (j < detections.size() && detections[j].class_id == detections[i].class_id) ++j;

        // [i, j) araligi tek sinif, confidence'e gore zaten sirali.
        std::vector<bool> suppressed(j - i, false);
        for (size_t a = i; a < j; ++a) {
            if (suppressed[a - i]) continue;
            result.push_back(detections[a]);

            for (size_t b = a + 1; b < j; ++b) {
                if (suppressed[b - i]) continue;
                // DIoU (plain IoU DEĞİL) — merkez mesafesi de cezalandırılır.
                float diou = calculateDiou(detections[a], detections[b]);
                if (diou > iou_thresh) {
                    suppressed[b - i] = true;
                }
            }
        }
        i = j;
    }

    std::sort(result.begin(), result.end(),
              [](const YoloDetection& a, const YoloDetection& b) {
                  return a.confidence > b.confidence;
              });
    return result;
}

std::vector<std::string> YoloPostProcessor::loadLabels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[YoloPostProcessor] Etiket dosyasi acilamadi: " << path << "\n";
        return labels;
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
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

// Referans: edge-ai-workshop-rknn/overlay.py _TRACK_COLORS ile AYNI 20
// renklik palet (Python tuple'lari cv2 konvansiyonuyla (B,G,R) — burada
// (R,G,B)'ye cevrilip RGA'nin (R<<16)|(G<<8)|B paketleme sirasina uyarlandi).
// get_color_for_track(): track_id % palet_boyu — her track HER ZAMAN ayni
// renk (confidence/kaybolma durumuna GORE DEGIL).
struct Rgb { uint8_t r, g, b; };
constexpr Rgb kTrackPalette[] = {
    {56, 56, 255},   {151, 157, 255}, {31, 112, 255},  {29, 178, 255},
    {49, 210, 207},  {10, 249, 72},   {23, 204, 146},  {134, 219, 61},
    {52, 147, 26},   {187, 212, 0},   {168, 153, 44},  {255, 194, 0},
    {147, 69, 52},   {255, 115, 100}, {236, 24, 0},    {255, 56, 132},
    {133, 0, 82},    {255, 56, 203},  {200, 149, 255}, {199, 55, 255},
};
constexpr int kTrackPaletteSize = sizeof(kTrackPalette) / sizeof(kTrackPalette[0]);

int colorForTrackId(int track_id) {
    int idx = track_id % kTrackPaletteSize;
    if (idx < 0) idx += kTrackPaletteSize;
    const Rgb& c = kTrackPalette[idx];
    return (static_cast<int>(c.r) << 16) | (static_cast<int>(c.g) << 8) | static_cast<int>(c.b);
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
// (RGA imfillArray, 4 ince kenar rect'i, 2-hizali). ByteTracker::update()
// sadece aktif track'leri dondurdugu icin buradaki liste = cizilecek liste;
// kaybolan nesne bir onceki karede listede yer almadigi icin ZATEN cizilmez
// (bekletme/soluklastirma YOK — kullanicinin "aninda kaybolsun" istegi).
void YoloPostProcessor::drawTrackedObjects(const DmaBufferPtr& frame,
                                            const std::vector<TrackedBox>& objects,
                                            int line_thickness) {
    if (!frame || frame->fd < 0 || objects.empty()) return;

    int fw = static_cast<int>(frame->width);
    int fh = static_cast<int>(frame->height);

    rga_buffer_t buf = wrapbuffer_fd(frame->fd, fw, fh, toRgaFormat(frame->format),
                                      static_cast<int>(frame->w_stride),
                                      static_cast<int>(frame->h_stride));

    static bool logged_error = false;

    for (const auto& obj : objects) {
        int x0 = std::max(0, static_cast<int>(obj.x));
        int y0 = std::max(0, static_cast<int>(obj.y));
        int x1 = std::min(fw, static_cast<int>(obj.x + obj.width));
        int y1 = std::min(fh, static_cast<int>(obj.y + obj.height));
        if (x1 <= x0 || y1 <= y0) continue;

        int t = std::min(line_thickness, std::min(x1 - x0, y1 - y0) / 2);
        if (t < 1) t = 1;

        int color = colorForTrackId(obj.track_id);

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
