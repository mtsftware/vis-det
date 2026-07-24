#pragma once

#include "buffer/DmaBufferPool.hpp"
#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>

// Referans: tracking-app/include/preprocess/RgaPreprocessor.hpp ile BİREBİR
// AYNI kalıp. Tek fark: tracking-app'teki cropTargetCentric() (siamese
// tracker search-region kırpması) ve flipHorizontal() (template augmentasyonu)
// burada YOK — YOLOv8n detector tam kareyi letterbox ile 640x640'a resize
// eder, hedef-merkezli kırpmaya ihtiyaç duymaz.
//
// KRİTİK MİMARİ KURALLAR (referansla aynı, Rapor 2/3):
// - immakeBorder() KESİNLİKLE KULLANILMAZ (RGA2 + DMA32/IOMMU çökme riski).
// - Letterbox iki RGA çağrısıyla kurulur: imfill (gri zemin, 114/114/114 —
//   YOLO'nun standart letterbox pad rengi) + improcess (ölçekle, ortala).
// - Kaynak formatı (NV12/NV16/...) ASLA varsayılmaz — decode'dan gelen
//   gerçek format çağırana açıkça parametre olarak verilir.
enum class RgaPixelFormat { NV12, NV16, RGB888, BGR888 };

struct LetterboxResult {
    double ratio = 1.0;
    int dst_offset_x = 0;
    int dst_offset_y = 0;
    int scaled_width = 0;
    int scaled_height = 0;
};

class RgaPreprocessor {
public:
    RgaPreprocessor();
    ~RgaPreprocessor();

    RgaPreprocessor(const RgaPreprocessor&) = delete;
    RgaPreprocessor& operator=(const RgaPreprocessor&) = delete;

    // target_format: RGB888 -> YOLO model girdisi (NHWC RGB, verification/read.md).
    bool configure(uint32_t target_width, uint32_t target_height,
                   RgaPixelFormat target_format);

    // Kaynağı (MPP decode çıktısı, NV12/NV16) aspect-ratio KORUYARAK
    // letterbox ile hedef boyuta (varsayılan 640x640) sığdırır. out_transform,
    // postprocess'in bbox'ları orijinal kareye geri map etmesi için gereken
    // ratio/offset bilgisini taşır (YoloPostProcessor::decodeOutputs bunu
    // kullanır — düz stretch/scale_x-scale_y YERİNE).
    bool process(const DmaBufferPtr& source, RgaPixelFormat source_format,
                 DmaBufferPtr& out_buffer, LetterboxResult& out_transform);

    // Kaynakla AYNI boyut/format'ta, havuzdan alınmış YENİ bir tampona
    // donanımsal (RGA improcess) piksel kopyası. Decode çıktısına yerinde
    // çizim yapmamak için (main'de çizim hedefi budur) — referanstaki
    // aynı gerekçe (bkz. tracking-app RgaPreprocessor.hpp cloneFrame notu).
    bool cloneFrame(const DmaBufferPtr& source, RgaPixelFormat source_format,
                     DmaBufferPtr& out_buffer);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};
