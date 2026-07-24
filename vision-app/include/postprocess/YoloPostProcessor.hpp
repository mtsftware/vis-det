#pragma once

#include "rga_processing/RgaPreprocessor.hpp"
#include "types/DmaBuffer.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

// YOLOv8n post-process, DIoU-NMS ve MOT çizimi.
// Model cikti: (1, 5, 8400) → channel 0-3: box, channel 4: score (tek-class,
// verification/read.md'de dogrulandi — skor kanali grafik icinde zaten
// Sigmoid'den gecmis, burada tekrar sigmoid UYGULANMAZ).
// Cikti 640x640 letterbox uzayinda — orijinal frame'e LetterboxResult ile
// (düz stretch/scale DEĞİL) map edilir; bkz. RgaPreprocessor::process().

struct YoloDetection {
    float x = 0;       // top-left x (orijinal frame piksel uzayi)
    float y = 0;       // top-left y
    float width = 0;   // box width
    float height = 0;  // box height
    float confidence = 0.0f;
    int class_id = 0;
    int track_id = -1;  // tracker tarafindan atanir
};

// Forward declare: MultiObjectTracker.hpp bu header'i (YoloDetection icin)
// include ediyor, dongusel include'dan kacinmak icin tam tanim yerine ileri
// bildirim yeterli (drawTrackedObjects sadece imza icin kullaniyor).
struct TrackableObject;

class YoloPostProcessor {
public:
    YoloPostProcessor();
    ~YoloPostProcessor() = default;

    // Model raw output'unu detection list'ine decode eder.
    //
    // raw_output: float32[1×5×8400] (want_float=1 ile RKNN'den gelen)
    // raw_size: bayt boyutu (normalde 1*5*8400*4 = 168000)
    // orig_w/orig_h: orijinal frame boyutlari (decode'dan gelen gercek boyut)
    // letterbox: RgaPreprocessor::process()'in urettigi ters-map bilgisi
    //            (ratio + dst_offset_x/y) — model 640x640 letterbox
    //            uzayindaki kutuyu orijinal kareye dogru geri tasir.
    // conf_thresh: minimum guven esigi (0.0~1.0)
    // out_max_score: non-null verilirse, esik uygulanmadan ONCE 8400
    //                anchor'in en yuksek skoru buraya yazilir — "0 tespit"
    //                durumunda modelin gercekten hicbir sey gormedigini mi
    //                yoksa skorlarin esigin biraz altinda mi kaldigini
    //                (veya girdi/decode bozuksa skorlarin anlamsiz/sabit
    //                kaldigini) ayirt etmek icin teshis amacli.
    //
    // Donus: confidence esigi uzerindeki tum tespitlerin listesi
    static bool decodeOutputs(const void* raw_output, uint32_t raw_size,
                              int orig_w, int orig_h,
                              const LetterboxResult& letterbox,
                              float conf_thresh,
                              std::vector<YoloDetection>& detections,
                              float* out_max_score = nullptr);

    // DIoU-NMS: birbiriyle örtüşen bbox'ları filtreler.
    // DIoU (Distance-IoU) — merkez mesafesini de cezalandirir, plain IoU'dan
    // daha kararli (özellikle yakin/örtüşen çoklu nesnelerde).
    // iou_thresh: esik (0.45~0.7 arasi onerilir)
    static std::vector<YoloDetection> applyNMS(std::vector<YoloDetection>& detections,
                                                 float iou_thresh = 0.45f);

    // Referans: tracking-app main_m11_test.cpp drawBboxOutline() ile AYNI
    // teknik (RGA imfillArray, 4 ince dikdortgen kenar, 2-hizali rect'ler) —
    // RGB donusumune GEREK YOK, dogrudan decode'un native formatinda (NV12/
    // NV16) frame uzerine cizer. Tek fark: referans TEK hedef ciziyordu,
    // burada TÜM aktif track'ler (MOT) donguyle cizilir, renk track_id'ye
    // gore degisir (kaybolan track'ler kirmizi/soluk).
    // frame: RgaPreprocessor::cloneFrame() cikisi (decode buffer'ina DEGIL,
    //        bagimsiz kopyaya cizilmeli).
    static void drawTrackedObjects(const DmaBufferPtr& frame,
                                    const std::vector<TrackableObject>& objects,
                                    int line_thickness = 4);

private:
    // Box alanı hesapla
    inline static float boxArea(const YoloDetection& d) {
        return d.width * d.height;
    }

    // DIoU (Distance-IoU) hesaplama
    inline static float calculateDiou(const YoloDetection& a, const YoloDetection& b) {
        // Intersection alanı
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.width, b.x + b.width);
        float y2 = std::min(a.y + a.height, b.y + b.height);

        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float union_area = boxArea(a) + boxArea(b) - intersection;

        if (union_area < 1e-6f) return 0.0f;

        float iou = intersection / union_area;

        // DIoU: center distance normalized by diagonal length
        float cx1 = a.x + a.width / 2.0f;
        float cy1 = a.y + a.height / 2.0f;
        float cx2 = b.x + b.width / 2.0f;
        float cy2 = b.y + b.height / 2.0f;

        float dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

        // Enclosing box diagonal
        float enc_x1 = std::min(a.x, b.x);
        float enc_y1 = std::min(a.y, b.y);
        float enc_x2 = std::max(a.x + a.width, b.x + b.width);
        float enc_y2 = std::max(a.y + a.height, b.y + b.height);
        float diag_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) + (enc_y2 - enc_y1) * (enc_y2 - enc_y1);

        if (diag_sq < 1e-6f) return iou;

        return iou - dist_sq / diag_sq;
    }
};
