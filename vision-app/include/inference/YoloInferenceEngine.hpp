#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// YOLOv8n RKNN model inference motoru.
//
// airockchip RKNN-optimised YOLOv8 export (edge-ai-workshop-rknn/inference.py
// ile AYNI model ailesi — su an yuklenen models/yolov8n_int8.rknn COCO
// 80-sinif). 9 CIKTI TENSORU:
//   box_s0, score_s0, sum_s0,   (stride 8,  H=W=80)
//   box_s1, score_s1, sum_s1,   (stride 16, H=W=40)
//   box_s2, score_s2, sum_s2    (stride 32, H=W=20)
//   box_s   : [1, 64, H, W]  4 kenar x REG_MAX=16 DFL bin, NCHW, float32
//   score_s : [1, nc, H, W]  sinif olasiliklari (sigmoid UYGULANMIS), NCHW
//   sum_s   : kullanilmiyor (postprocess'te atlanir)
// Girdi: 640x640x3 NHWC UINT8 (pass_through=0, RKNN runtime int8 kuantize eder).
//
// eski best-rk3588.rknn (tek-tensor, DFL grafige gomulu, tek-sinif) icin
// ayri bir motor gerekmiyordu; bu siniftaki API artik N-cikti genel amacli
// (outputCount()/getOutput(index,...)) — tek-tensor modeller icin
// outputCount()==1 ile ayni sekilde calisir.
//
// Referans: tracking-app/src/inference/RknnInferenceEngine.cpp ile ayni
// librknnrt.so kullanir, ama YOLO icin optimize edilmis API sunar.

class YoloInferenceEngine {
public:
    YoloInferenceEngine();
    ~YoloInferenceEngine();

    YoloInferenceEngine(const YoloInferenceEngine&) = delete;
    YoloInferenceEngine& operator=(const YoloInferenceEngine&) = delete;

    // Model (.rknn) yukler. Cagirlandan sonra inference hazir olur.
    bool loadModel(const std::string& rknn_model_path);

    // Inference calistirir.
    // input_buf: RGA cikti (640x640x3 uint8 RGB), DOGRUDAN isaretci verilir
    //            (kopya ALINMAZ — zero-copy). run() donene kadar live kalmali.
    //
    // Donus: true = basarili, false = hata (log bakiniz).
    bool run(const void* input_buf, uint32_t input_size);

    // TUM cikti tensorlerini TEK rknn_outputs_get cagrisiyla getirir (RKNN
    // cok-cikti modellerde ayri ayri degil, hepsini bir arada istemeyi
    // bekler — bkz. rknn_api.h). want_float=1: runtime dequant (float32).
    // run()'dan SONRA, getOutput() cagirmadan ONCE cagrilmali.
    bool fetchOutputs();

    // Model yuklendikten sonra kac cikti tensoru oldugunu doner (bu model
    // icin 9; eski tek-tensor modeller icin 1).
    int outputCount() const;

    // fetchOutputs() sonrasi index'inci cikti tensorunu okur.
    // output_data: float32 buffer, output_size: bayt boyutu.
    bool getOutput(int index, const void*& output_data, uint32_t& output_size) const;

    // Ciktlari serbest birak (rknn_outputs_release) — sonraki run() once otomatik
    // cagrilir ama manuel kontroll icin de sunilir.
    void releaseOutputs();

    // Model yuklendikten sonra input/output profil bilgilerini doner.
    struct TensorInfo {
        int dims[4] = {0, 0, 0, 0};
        int n_dims = 0;
        uint32_t size = 0;
        uint32_t n_elems = 0;
        int zp = 0;
        float scale = 1.0f;
        int fmt = 0;  // rknn_tensor_format ham degeri (0=NCHW, 1=NHWC, ...)
        std::string name;
    };
    const TensorInfo& inputInfo() const;
    const TensorInfo& outputInfo(int index) const;

    // NPU inference suresi (mikrosaniye) — run()+getOutput() cagrilarindan SONRA gecerli.
    int64_t lastRunDurationUs() const;

    void unload();

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};
