#pragma once

#include "inference/YoloInferenceEngine.hpp"
#include "rga_processing/RgaPreprocessor.hpp"
#include "types/DmaBuffer.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

// YOLOv8n (airockchip RKNN-optimised export) post-process, DIoU-NMS ve MOT
// cizimi. Referans: edge-ai-workshop-rknn/inference.py (_postprocess_rknn,
// _dfl_decode) — AYNI model ailesi (models/yolov8n_int8.rknn, COCO 80-sinif).
//
// Model 9 cikti tensoru verir (eski tek-tensor best-rk3588.rknn DEGIL):
//   box_s0,score_s0,sum_s0 (stride 8, 80x80), box_s1,score_s1,sum_s1
//   (stride 16, 40x40), box_s2,score_s2,sum_s2 (stride 32, 20x20).
//   box_s   : [1,64,H,W] NCHW  4 kenar x REG_MAX=16 DFL dagilimi
//   score_s : [1,nc,H,W] NCHW  sinif olasiliklari (sigmoid UYGULANMIS,
//             burada TEKRAR sigmoid UYGULANMAZ)
// DFL decode + letterbox ters-map sonucu kutu, orijinal frame piksel
// uzayinda (x,y,width,height) olarak YoloDetection'a yazilir.

struct YoloDetection {
    float x = 0;       // top-left x (orijinal frame piksel uzayi)
    float y = 0;       // top-left y
    float width = 0;   // box width
    float height = 0;  // box height
    float confidence = 0.0f;
    int class_id = 0;   // COCO sinif indeksi (coco_labels.txt satir no)
    int track_id = -1;  // tracker tarafindan atanir
};

// Forward declare: ByteTracker.hpp bu header'i (YoloDetection icin) include
// ediyor, dongusel include'dan kacinmak icin tam tanim yerine ileri bildirim
// yeterli (drawTrackedObjects sadece imza icin kullaniyor).
struct TrackedBox;

class YoloPostProcessor {
public:
    YoloPostProcessor();
    ~YoloPostProcessor() = default;

    // engine.fetchOutputs() cagrildiktan SONRA cagrilmali (9 tensoru okur).
    //
    // orig_w/orig_h: orijinal frame boyutlari (decode'dan gelen gercek boyut)
    // letterbox: RgaPreprocessor::process()'in urettigi ters-map bilgisi
    //            (ratio + dst_offset_x/y) — model 640x640 letterbox
    //            uzayindaki kutuyu orijinal kareye dogru geri tasir.
    // conf_thresh: minimum guven esigi (0.0~1.0)
    // out_max_score: non-null verilirse, esik uygulanmadan ONCE tum
    //                anchor'lar arasindaki en yuksek skor buraya yazilir —
    //                teshis amacli (bkz. "0 tespit" arastirmasi).
    //
    // Donus: confidence esigi uzerindeki tum tespitlerin listesi
    static bool decodeOutputs(const YoloInferenceEngine& engine,
                              int orig_w, int orig_h,
                              const LetterboxResult& letterbox,
                              float conf_thresh,
                              std::vector<YoloDetection>& detections,
                              float* out_max_score = nullptr);

    // DIoU-NMS: HER SINIF ICINDE AYRI uygulanir (farkli siniflarin ust uste
    // binen kutulari birbirini bastirmaz) — referans: inference.py
    // _postprocess_rknn()'deki per-class NMS dongusu.
    // iou_thresh: esik (0.45~0.7 arasi onerilir)
    static std::vector<YoloDetection> applyNMS(std::vector<YoloDetection>& detections,
                                                 float iou_thresh = 0.45f);

    // coco_labels.txt (satir basina bir isim) yukler. Bulunamazsa bos
    // liste doner (cagiran numerik ID'ye geri duser).
    static std::vector<std::string> loadLabels(const std::string& path);

    // Cizim TEKNIGI referansi: tracking-app main_m11_test.cpp
    // drawBboxOutline() ile AYNI (RGA imfillArray, 4 ince dikdortgen kenar,
    // 2-hizali rect'ler) — RGB donusumune GEREK YOK, dogrudan decode'un
    // native formatinda (NV12/NV16) frame uzerine cizer.
    // RENK semasi referansi: edge-ai-workshop-rknn/overlay.py
    // get_color_for_track() ile AYNI 20 renklik sabit palet — her track_id
    // HER ZAMAN ayni renkte (confidence/kaybolma durumuna gore DEGIL, RGA
    // metin cizemedigi icin ID rozeti yerine renk kararliligi kullanilir).
    // ByteTracker::update() zaten SADECE aktif (Tracked) track'leri
    // dondurdugu icin (bkz. ByteTracker.hpp), burada "kaybolan" durumu YOK —
    // liste ne veriyorsa o cizilir, kaybolan nesne bir SONRAKI karede
    // listede hic olmadigi icin ANINDA cizimden duser.
    // frame: RgaPreprocessor::cloneFrame() cikisi (decode buffer'ina DEGIL,
    //        bagimsiz kopyaya cizilmeli).
    static void drawTrackedObjects(const DmaBufferPtr& frame,
                                    const std::vector<TrackedBox>& objects,
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
