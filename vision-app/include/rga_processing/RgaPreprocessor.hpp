#pragma once

#include "buffer/DmaBufferPool.hpp"
#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>
#include <string>

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};