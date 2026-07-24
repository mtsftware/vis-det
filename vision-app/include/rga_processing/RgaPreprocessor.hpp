#pragma once

#include "buffer/DmaBufferPool.hpp"
#include "types/DmaBuffer.hpp"
#include "tracking/MultiObjectTracker.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// RGA katmanina ozgu piksel format enum (referans tracking-app ile ayni)
enum class RgaPixelFormat { NV12, NV16, RGB888, BGR888 };

class RgaPreprocessor {
public:
    RgaPreprocessor();
    ~RgaPreprocessor();

    RgaPreprocessor(const RgaPreprocessor&) = delete;
    RgaPreprocessor& operator=(const RgaPreprocessor&) = delete;

    // NV12 (src) -> RGB888 (dst) dönüşüm (format + scale bir işlemte)
    bool nv12ToRgb888(const DmaBufferPtr& src_nv12, uint32_t src_w, uint32_t src_h,
                      DmaBufferPtr& dst_rgb, uint32_t dst_w, uint32_t dst_h);

    // NV16 (src) -> RGB888 (dst) dönüşüm (format + scale bir işlemte)
    bool nv16ToRgb888(const DmaBufferPtr& src_nv16, uint32_t src_w, uint32_t src_h,
                      DmaBufferPtr& dst_rgb, uint32_t dst_w, uint32_t dst_h);

    // Otomatik format algılama: NV12 veya NV16 input kabul eder, RGB888 output verir
    bool processYuvToRgb888(const DmaBufferPtr& src_yuv, PixelFormat src_fmt,
                            uint32_t src_w, uint32_t src_h,
                            DmaBufferPtr& dst_rgb, uint32_t dst_w, uint32_t dst_h);

    // RGB888 buffer uzerine 3 adet dummy BBox cizer (kirmizi/yesil/mavi).
    bool draw3DummyBboxes(DmaBufferPtr& rgb_buffer, uint32_t w, uint32_t h);

    // Tespit edilen object'leri class-bazli renklerle cizer + ID etiketi ekler.
    // renkler: insan=yeþil(0,255,0), araç=mavi(255,0,0) varsayýlýr.
    // line_thickness: çizgi kalýnlýðý (piksel).
    // draw_id_label: true ise bbox üzerine ID yaz (basit text rectangle).
    bool drawDetectedObjects(DmaBufferPtr& rgb_buffer, uint32_t w, uint32_t h,
                             const std::vector<TrackableObject>& objects,
                             int line_thickness = 2,
                             bool draw_id_label = true);

    // RGB888 -> NV12 dönüşüm + resize (encoder icin geri çevrim)
    bool rgb888ToNv12(const DmaBufferPtr& src_rgb, uint32_t src_w, uint32_t src_h,
                      DmaBufferPtr& dst_nv12, uint32_t dst_w, uint32_t dst_h);

    // RGB888 -> NV16 dönüşüm + resize (encoder icin geri çevrim)
    bool rgb888ToNv16(const DmaBufferPtr& src_rgb, uint32_t src_w, uint32_t src_h,
                      DmaBufferPtr& dst_nv16, uint32_t dst_w, uint32_t dst_h);

    // Otomatik format algılama: RGB888 input kabul eder, NV12 veya NV16 output verir
    bool processRgb888ToYuv(const DmaBufferPtr& src_rgb, uint32_t src_w, uint32_t src_h,
                            PixelFormat dst_fmt,
                            DmaBufferPtr& dst_yuv, uint32_t dst_w, uint32_t dst_h);

    // RGB888 buffer'i diske PPM3 (renkli) olarak kaydeder
    static bool saveRgb888ToPpm(const DmaBufferPtr& rgb_buffer, const std::string& path,
                                 uint32_t w, uint32_t h);

    // RGA ciktisi icin havuzdan buffer tahsisi (DMA fd ile)
    DmaBufferPtr acquireOutputBuffer(uint32_t bytes, uint32_t w, uint32_t h,
                                      PixelFormat px_fmt);

    // Kaynak buffer'in tamamen ayri bir kopyasini olusturur.
    // Decoder buffer'ina yerinde yazma (çizim vb.) onlemek icin kullanilir.
    // pikselkaymasi.md §3 Adimlar ile uyumlu implementasyon.
    bool cloneFrame(const DmaBufferPtr& source, RgaPixelFormat source_format,
                    DmaBufferPtr& out_buffer);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};