#include "rga_processing/RgaPreprocessor.hpp"

#include "im2d.h"
#include "rga.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

struct RgaPreprocessor::Impl {
    DmaBufferPool pool;

    // Havuzdan buffer al (RGA ciktisi icin)
    DmaBufferPtr acquireBuffer(const char* tag, size_t bytes, uint32_t w, uint32_t h,
                                PixelFormat px_fmt) {
        std::string name = std::string(tag) + std::to_string(bytes);
        if (!pool.ensurePool(name, static_cast<uint32_t>(bytes), 4, HeapKind::CachedDma32)) {
            std::cerr << "[RgaPreprocessor] havuz kurulamadı: " << name << "\n";
            return nullptr;
        }
        DmaBufferPtr buf = pool.tryAcquireFor(name, 500);
        if (!buf) {
            std::cerr << "[RgaPreprocessor] havuz zaman aşımı: " << name << "\n";
            return nullptr;
        }
        buf->width = w;
        buf->height = h;
        buf->w_stride = w;
        buf->h_stride = h;
        buf->format = px_fmt;
        buf->dtype = DataType::UINT8;
        return buf;
    }
};

RgaPreprocessor::RgaPreprocessor() : impl_(std::make_unique<Impl>()) {}
RgaPreprocessor::~RgaPreprocessor() = default;

DmaBufferPtr RgaPreprocessor::acquireOutputBuffer(uint32_t bytes, uint32_t w, uint32_t h,
                                                   PixelFormat px_fmt) {
    return impl_->acquireBuffer("out-", bytes, w, h, px_fmt);
}

namespace {

static uint32_t align16(uint32_t v) {
    return ((v + 15) / 16) * 16;
}

static int toRgaFormat(RgaPixelFormat fmt) {
    switch (fmt) {
        case RgaPixelFormat::NV12:
            return RK_FORMAT_YCbCr_420_SP;
        case RgaPixelFormat::NV16:
            return RK_FORMAT_YCbCr_422_SP;
        case RgaPixelFormat::RGB888:
            return RK_FORMAT_RGB_888;
        case RgaPixelFormat::BGR888:
            return RK_FORMAT_BGR_888;
    }
    return RK_FORMAT_YCbCr_420_SP;
}

static size_t formatByteSize(RgaPixelFormat fmt, uint32_t w, uint32_t h) {
    switch (fmt) {
        case RgaPixelFormat::NV12:
            return static_cast<size_t>(w) * h * 3 / 2;
        case RgaPixelFormat::NV16:
            return static_cast<size_t>(w) * h * 2;
        case RgaPixelFormat::RGB888:
        case RgaPixelFormat::BGR888:
            return static_cast<size_t>(w) * h * 3;
    }
    return static_cast<size_t>(w) * h * 3 / 2;
}

}  // namespace

bool RgaPreprocessor::nv12ToRgb888(const DmaBufferPtr& src_nv12, uint32_t src_w,
                                    uint32_t src_h, DmaBufferPtr& dst_rgb,
                                    uint32_t dst_w, uint32_t dst_h) {
    return processYuvToRgb888(src_nv12, PixelFormat::NV12, src_w, src_h, dst_rgb, dst_w, dst_h);
}

bool RgaPreprocessor::nv16ToRgb888(const DmaBufferPtr& src_nv16, uint32_t src_w,
                                    uint32_t src_h, DmaBufferPtr& dst_rgb,
                                    uint32_t dst_w, uint32_t dst_h) {
    return processYuvToRgb888(src_nv16, PixelFormat::NV16, src_w, src_h, dst_rgb, dst_w, dst_h);
}

bool RgaPreprocessor::processYuvToRgb888(const DmaBufferPtr& src_yuv, PixelFormat src_fmt,
                                          uint32_t src_w, uint32_t src_h,
                                          DmaBufferPtr& dst_rgb, uint32_t dst_w, uint32_t dst_h) {
    if (!src_yuv || src_yuv->fd < 0 || !dst_rgb) return false;

    int src_rga_fmt;
    if (src_fmt == PixelFormat::NV16) {
        src_rga_fmt = RK_FORMAT_YCbCr_422_SP;  // NV16
    } else {
        src_rga_fmt = RK_FORMAT_YCbCr_420_SP;  // NV12
    }
    int dst_rga_fmt = RK_FORMAT_RGB_888;

    rga_buffer_t src_buf = wrapbuffer_fd(src_yuv->fd, static_cast<int>(src_w),
                                          static_cast<int>(src_h), src_rga_fmt,
                                          static_cast<int>(src_yuv->w_stride),
                                          static_cast<int>(src_yuv->h_stride));

    uint32_t dst_stride = align16(dst_w);
    uint32_t dst_h_stride = align16(dst_h);  // Height hizalama (RK3588 chroma padding)
    rga_buffer_t dst_buf = wrapbuffer_fd(dst_rgb->fd, static_cast<int>(dst_w),
                                          static_cast<int>(dst_h), dst_rga_fmt,
                                          static_cast<int>(dst_stride),
                                          static_cast<int>(dst_h_stride));

    im_rect src_rect = {0, 0, static_cast<int>(src_w), static_cast<int>(src_h)};
    im_rect dst_rect = {0, 0, static_cast<int>(dst_w), static_cast<int>(dst_h)};

    IM_STATUS check_ret = imcheck(src_buf, dst_buf, src_rect, dst_rect);
    if (check_ret != IM_STATUS_NOERROR) {
        std::cerr << "[RgaPreprocessor] processYuvToRgb888 imcheck basarisiz ("
                  << (src_fmt == PixelFormat::NV16 ? "NV16" : "NV12")
                  << "): " << imStrError(check_ret) << "\n";
        return false;
    }

    rga_buffer_t empty_pat{};
    im_rect empty_rect{};
    IM_STATUS proc_ret = improcess(src_buf, dst_buf, empty_pat, src_rect, dst_rect,
                                    empty_rect, 0);
    if (proc_ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RgaPreprocessor] processYuvToRgb888 improcess basarisiz: "
                  << imStrError(proc_ret) << "\n";
        return false;
    }

    impl_->pool.syncCpuReadBegin(dst_rgb);

    const char* fmt_name = (src_fmt == PixelFormat::NV16) ? "NV16" : "NV12";
    std::cout << "[RgaPreprocessor] " << fmt_name << "->RGB888 donusum tamamlandi: "
              << src_w << "x" << src_h << " -> " << dst_w << "x" << dst_h << "\n";
    return true;
}

bool RgaPreprocessor::draw3DummyBboxes(DmaBufferPtr& rgb_buffer, uint32_t w, uint32_t h) {
    if (!rgb_buffer || rgb_buffer->fd < 0) return false;

    int rga_fmt = RK_FORMAT_RGB_888;
    uint32_t stride = align16(w);

    rga_buffer_t img_buf = wrapbuffer_fd(rgb_buffer->fd, static_cast<int>(w),
                                          static_cast<int>(h), rga_fmt,
                                          static_cast<int>(stride), static_cast<int>(h));

    struct {
        int x, y, width, height;
        int color;
    } bboxes[] = {
        {10, 10, 100, 80, (255 << 16) | (0 << 8) | 0},  // Kirmizi
        {static_cast<int>(w/2) - 50, static_cast<int>(h/2) - 40, 100, 80,
         (0 << 16) | (255 << 8) | 0},                    // Yesil
        {static_cast<int>(w) - 120, static_cast<int>(h) - 100, 100, 80,
         (0 << 16) | (0 << 8) | 255},                    // Mavi
    };

    int line_thickness = 2;
    for (const auto& bbox : bboxes) {
        im_rect top_rect = {bbox.x, bbox.y, bbox.width, line_thickness};
        imfill(img_buf, top_rect, bbox.color);
        im_rect bottom_rect = {bbox.x, bbox.y + bbox.height - line_thickness,
                               bbox.width, line_thickness};
        imfill(img_buf, bottom_rect, bbox.color);
        im_rect left_rect = {bbox.x, bbox.y, line_thickness, bbox.height};
        imfill(img_buf, left_rect, bbox.color);
        im_rect right_rect = {bbox.x + bbox.width - line_thickness, bbox.y,
                              line_thickness, bbox.height};
        imfill(img_buf, right_rect, bbox.color);
    }

    std::cout << "[RgaPreprocessor] 3 adet dummy BBox cizildi.\n";
    return true;
}

bool RgaPreprocessor::rgb888ToNv12(const DmaBufferPtr& src_rgb, uint32_t src_w,
                                    uint32_t src_h, DmaBufferPtr& dst_nv12,
                                    uint32_t dst_w, uint32_t dst_h) {
    return processRgb888ToYuv(src_rgb, src_w, src_h, PixelFormat::NV12, dst_nv12, dst_w, dst_h);
}

bool RgaPreprocessor::rgb888ToNv16(const DmaBufferPtr& src_rgb, uint32_t src_w,
                                    uint32_t src_h, DmaBufferPtr& dst_nv16,
                                    uint32_t dst_w, uint32_t dst_h) {
    return processRgb888ToYuv(src_rgb, src_w, src_h, PixelFormat::NV16, dst_nv16, dst_w, dst_h);
}

bool RgaPreprocessor::processRgb888ToYuv(const DmaBufferPtr& src_rgb, uint32_t src_w,
                                          uint32_t src_h, PixelFormat dst_fmt,
                                          DmaBufferPtr& dst_yuv, uint32_t dst_w, uint32_t dst_h) {
    if (!src_rgb || src_rgb->fd < 0 || !dst_yuv) return false;

    int src_rga_fmt = RK_FORMAT_RGB_888;
    int dst_rga_fmt;
    if (dst_fmt == PixelFormat::NV16) {
        dst_rga_fmt = RK_FORMAT_YCbCr_422_SP;  // NV16
    } else {
        dst_rga_fmt = RK_FORMAT_YCbCr_420_SP;  // NV12
    }

    uint32_t src_stride = align16(src_w);
    uint32_t dst_stride = align16(dst_w);
    uint32_t dst_h_stride = align16(dst_h);  // Height hizalama (RK3588 chroma padding)

    rga_buffer_t src_buf = wrapbuffer_fd(src_rgb->fd, static_cast<int>(src_w),
                                          static_cast<int>(src_h), src_rga_fmt,
                                          static_cast<int>(src_stride),
                                          static_cast<int>(src_h));

    rga_buffer_t dst_buf = wrapbuffer_fd(dst_yuv->fd, static_cast<int>(dst_w),
                                          static_cast<int>(dst_h), dst_rga_fmt,
                                          static_cast<int>(dst_stride),
                                          static_cast<int>(dst_h_stride));

    im_rect src_rect = {0, 0, static_cast<int>(src_w), static_cast<int>(src_h)};
    im_rect dst_rect = {0, 0, static_cast<int>(dst_w), static_cast<int>(dst_h)};

    IM_STATUS check_ret = imcheck(src_buf, dst_buf, src_rect, dst_rect);
    if (check_ret != IM_STATUS_NOERROR) {
        std::cerr << "[RgaPreprocessor] processRgb888ToYuv imcheck basarisiz: "
                  << imStrError(check_ret) << "\n";
        return false;
    }

    rga_buffer_t empty_pat{};
    im_rect empty_rect{};
    IM_STATUS proc_ret = improcess(src_buf, dst_buf, empty_pat, src_rect, dst_rect,
                                    empty_rect, 0);
    if (proc_ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RgaPreprocessor] processRgb888ToYuv improcess basarisiz: "
                  << imStrError(proc_ret) << "\n";
        return false;
    }

    const char* fmt_name = (dst_fmt == PixelFormat::NV16) ? "NV16" : "NV12";
    std::cout << "[RgaPreprocessor] RGB888->" << fmt_name
              << " donusum tamamlandi: " << src_w << "x" << src_h
              << " -> " << dst_w << "x" << dst_h << "\n";
    return true;
}

bool RgaPreprocessor::cloneFrame(const DmaBufferPtr& source, RgaPixelFormat source_format,
                                  DmaBufferPtr& out_buffer) {
    // pikselkaymasi.md §3 Adim 1: Ön koşul kontrolü
    if (!source || source->fd < 0) {
        return false;
    }

    uint32_t w = source->width;
    uint32_t h = source->height;

    // pikselkaymasi.md §3 Adim 2: Boyut/format belirle, byte hesapla
    // Hedef her zaman SIKI-PAKETLI (w_stride=w, h_stride=h) — dolgu payi yok
    size_t bytes = formatByteSize(source_format, w, h);
    PixelFormat px = (source_format == RgaPixelFormat::NV16) ? PixelFormat::NV16
                                                              : PixelFormat::NV12;

    // pikselkaymasi.md §3 Adim 3: Havuzdan buffer al
    DmaBufferPtr buf = impl_->acquireBuffer("frame-clone-", static_cast<uint32_t>(bytes), w, h, px);
    if (!buf) {
        return false;
    }

    // pikselkaymasi.md §3 Adim 4: RGA wrapper'larini hazirla
    // Kaynak: gerçek stride ile, Hedef: tightly-packed (stride argümani verilmez)
    int rga_fmt = toRgaFormat(source_format);
    rga_buffer_t src_buf = wrapbuffer_fd(source->fd, static_cast<int>(w),
                                          static_cast<int>(h), rga_fmt,
                                          static_cast<int>(source->w_stride),
                                          static_cast<int>(source->h_stride));
    rga_buffer_t dst_buf =
        wrapbuffer_fd(buf->fd, static_cast<int>(w), static_cast<int>(h), rga_fmt);

    // pikselkaymasi.md §3 Adim 5: imcheck ile dogrulama, sonra improcess ile kopyala
    im_rect rect = {0, 0, static_cast<int>(w), static_cast<int>(h)};
    IM_STATUS check_ret = imcheck(src_buf, dst_buf, rect, rect);
    if (check_ret != IM_STATUS_NOERROR) {
        std::cerr << "[RgaPreprocessor] cloneFrame imcheck basarisiz: "
                  << imStrError(check_ret) << "\n";
        return false;
    }

    rga_buffer_t empty_pat{};
    im_rect empty_rect{};
    IM_STATUS proc_ret = improcess(src_buf, dst_buf, empty_pat, rect, rect, empty_rect, 0);
    if (proc_ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RgaPreprocessor] cloneFrame improcess basarisiz: "
                  << imStrError(proc_ret) << "\n";
        return false;
    }

    // pikselkaymasi.md §3 Adim 6: Cache senkronizasyonu, sonucu döndür
    impl_->pool.syncCpuReadBegin(buf);
    out_buffer = buf;
    return true;
}

bool RgaPreprocessor::saveRgb888ToPpm(const DmaBufferPtr& rgb_buffer, const std::string& path,
                                       uint32_t w, uint32_t h) {
    if (!rgb_buffer || !rgb_buffer->virt_addr) return false;

    const uint8_t* data = static_cast<const uint8_t*>(rgb_buffer->virt_addr);
    size_t expected_size = static_cast<size_t>(w) * h * 3;
    if (rgb_buffer->size < expected_size) {
        std::cerr << "[RgaPreprocessor] PPM kaydetme icin buffer cok kucuk\n";
        return false;
    }

    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        std::cerr << "[RgaPreprocessor] Dosya acilamadi: " << path << "\n";
        return false;
    }

    std::ostringstream header;
    header << "P3\n" << w << " " << h << "\n255\n";
    fout.write(header.str().c_str(), static_cast<std::streamsize>(header.str().size()));

    for (uint32_t i = 0; i < w * h; ++i) {
        int r = data[i * 3];
        int g = data[i * 3 + 1];
        int b = data[i * 3 + 2];
        fout << r << " " << g << " " << b << " ";
    }
    fout.close();

    std::cout << "[RgaPreprocessor] Renkli goruntu kaydedildi: " << path << "\n";
    return true;
}