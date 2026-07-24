#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// YOLOv8n RKNN model inference motoru.
// Model yapisi (cihazda dogrulandi - verification/read.md):
//   Input:  640x640x3 NHWC INT8 (pixel-128 → int8)
//   Output: (1, 5, 8400) — 8400 anchor, 4 box + 1 score (nc=1)
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

    // Cikti tensor'u okur. want_float=1 ise RKNN runtime dequant yapar (float32).
    // output_data: donen float32 buffer (1×5×8400 × 4 byte = 168.000 byte)
    // output_size: buffer buyuklugu
    bool getOutput(const void*& output_data, uint32_t& output_size);

    // Ciktlari serbest birak (rknn_outputs_release) — sonraki run() once otomatik
    // cagrilir ama manuel kontroll icin de sunilir.
    void releaseOutputs();

    // Model yuklendikten sonra input/output profil bilgilerini doner.
    struct TensorInfo {
        int dims[4] = {0, 0, 0, 0};
        int n_dims = 0;
        uint32_t size = 0;
        int zp = 0;
        float scale = 1.0f;
        std::string name;
    };
    const TensorInfo& inputInfo() const;
    const TensorInfo& outputInfo() const;

    // NPU inference suresi (mikrosaniye) — run()+getOutput() cagrilarindan SONRA gecerli.
    int64_t lastRunDurationUs() const;

    void unload();

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};