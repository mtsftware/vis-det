#include "rga_processing/RgaPreprocessor.hpp"

#include "im2d.h"
#include "rga.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

// Referans: tracking-app/src/preprocess/RgaPreprocessor.cpp ile BİREBİR AYNI
// kalıp (configure/process/cloneFrame + Impl::acquireBuffer havuz deseni).
// cropTargetCentric()/flipHorizontal() (siamese tracker'a özgü) yok — bkz.
// header notu.

namespace {

int toRgaFormat(RgaPixelFormat fmt) {
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

size_t formatByteSize(RgaPixelFormat fmt, uint32_t w, uint32_t h) {
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

// Havuz yönetimi referansla aynı: RgaPreprocessor kendi buffer'ını TAHSİS
// ETMEZ, DmaBufferPool'dan ödünç alır; iade shared_ptr referans sayısı
// sıfıra inince otomatik (custom deleter, bkz. 04-IBufferPool.md).
struct RgaPreprocessor::Impl {
    uint32_t target_width = 0;
    uint32_t target_height = 0;
    RgaPixelFormat target_format = RgaPixelFormat::RGB888;
    bool configured = false;

    DmaBufferPool pool;

    DmaBufferPtr acquireBuffer(const char* tag, size_t bytes, uint32_t w, uint32_t h,
                                PixelFormat px_fmt) {
        std::string name = std::string(tag) + std::to_string(bytes);
        if (!pool.ensurePool(name, static_cast<uint32_t>(bytes), 4, HeapKind::CachedDma32)) {
            std::cerr << "[RgaPreprocessor] havuz kurulamadı: " << name << "\n";
            return nullptr;
        }
        DmaBufferPtr buf = pool.tryAcquireFor(name, 500);
        if (!buf) {
            std::cerr << "[RgaPreprocessor] havuz zaman aşımı (" << name
                      << ") — tüketici buffer'ları iade etmiyor olabilir\n";
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

bool RgaPreprocessor::configure(uint32_t target_width, uint32_t target_height,
                                 RgaPixelFormat target_format) {
    // 16-byte hizalama kuralı (Rapor 2 & 3): RGA3 üzerinden RGB/YUV
    // çıktılarda genişlik hizası zorunlu.
    impl_->target_width = ((target_width + 15) / 16) * 16;
    impl_->target_height = target_height;
    impl_->target_format = target_format;
    impl_->configured = true;

    if (impl_->target_width != target_width) {
        std::cout << "[RgaPreprocessor] Genişlik " << target_width << " -> "
                  << impl_->target_width << " olarak 16-byte hizalandı.\n";
    }
    return true;
}

bool RgaPreprocessor::process(const DmaBufferPtr& source, RgaPixelFormat source_format,
                               DmaBufferPtr& out_buffer, LetterboxResult& out_transform) {
    if (!impl_->configured || !source || source->fd < 0) {
        return false;
    }

    const uint32_t tw = impl_->target_width;
    const uint32_t th = impl_->target_height;

    // Letterbox oranı: aspect-ratio korunarak hedefe sığdırma (Rapor 3) —
    // YOLOv8n'in eğitildiği letterbox preprocessing'iyle AYNI mantık.
    double ratio = std::min(static_cast<double>(tw) / source->width,
                             static_cast<double>(th) / source->height);
    int scaled_w = static_cast<int>(source->width * ratio);
    int scaled_h = static_cast<int>(source->height * ratio);
    int offset_x = (static_cast<int>(tw) - scaled_w) / 2;
    int offset_y = (static_cast<int>(th) - scaled_h) / 2;

    out_transform.ratio = ratio;
    out_transform.dst_offset_x = offset_x;
    out_transform.dst_offset_y = offset_y;
    out_transform.scaled_width = scaled_w;
    out_transform.scaled_height = scaled_h;

    size_t dst_size = formatByteSize(impl_->target_format, tw, th);
    PixelFormat out_px = (impl_->target_format == RgaPixelFormat::NV12)
                              ? PixelFormat::NV12
                              : PixelFormat::RGB888;
    DmaBufferPtr buf = impl_->acquireBuffer("letterbox-", dst_size, tw, th, out_px);
    if (!buf) return false;

    int src_rga_fmt = toRgaFormat(source_format);
    int dst_rga_fmt = toRgaFormat(impl_->target_format);

    rga_buffer_t src_buf =
        wrapbuffer_fd(source->fd, static_cast<int>(source->width),
                      static_cast<int>(source->height), src_rga_fmt,
                      static_cast<int>(source->w_stride),
                      static_cast<int>(source->h_stride));
    rga_buffer_t dst_buf = wrapbuffer_fd(buf->fd, static_cast<int>(tw),
                                          static_cast<int>(th), dst_rga_fmt);

    im_rect full_dst_rect = {0, 0, static_cast<int>(tw), static_cast<int>(th)};
    im_rect src_rect = {0, 0, static_cast<int>(source->width),
                         static_cast<int>(source->height)};
    im_rect dst_rect = {offset_x, offset_y, scaled_w, scaled_h};

    // ADIM 1: Zemini gri ile doldur (114,114,114 — YOLO letterbox pad rengi;
    // immakeBorder KESİNLİKLE KULLANILMAZ, bkz. header notu).
    int gray_color = (114 << 16) | (114 << 8) | 114;
    IM_STATUS fill_ret = imfill(dst_buf, full_dst_rect, gray_color);
    if (fill_ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RgaPreprocessor] imfill başarısız: " << imStrError(fill_ret)
                  << "\n";
    }

    // ADIM 2: Kaynağı ölçekle, format dönüştür, merkeze yerleştir.
    rga_buffer_t empty_pat{};
    im_rect empty_rect{};

    IM_STATUS check_ret = imcheck(src_buf, dst_buf, src_rect, dst_rect);
    if (check_ret != IM_STATUS_NOERROR) {
        std::cerr << "[RgaPreprocessor] imcheck başarısız: " << imStrError(check_ret)
                  << "\n";
        return false;  // buf otomatik havuza döner (RAII)
    }

    IM_STATUS proc_ret =
        improcess(src_buf, dst_buf, empty_pat, src_rect, dst_rect, empty_rect, 0);
    if (proc_ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RgaPreprocessor] improcess başarısız: " << imStrError(proc_ret)
                  << "\n";
        return false;
    }

    // CPU okuyacaksa (rknn_inputs_set) cache invalidate gerekli.
    impl_->pool.syncCpuReadBegin(buf);

    out_buffer = buf;
    return true;
}

bool RgaPreprocessor::cloneFrame(const DmaBufferPtr& source, RgaPixelFormat source_format,
                                  DmaBufferPtr& out_buffer) {
    if (!source || source->fd < 0) return false;

    uint32_t w = source->width;
    uint32_t h = source->height;

    // Hedef her zaman SIKI-PAKETLİ (w_stride=w, h_stride=h). Kaynağın kendi
    // stride'ı (dolgu payı olabilir) src_buf'a ayrıca veriliyor.
    size_t bytes = formatByteSize(source_format, w, h);
    PixelFormat px = (source_format == RgaPixelFormat::NV16) ? PixelFormat::NV16
                                                              : PixelFormat::NV12;
    DmaBufferPtr buf = impl_->acquireBuffer("frame-clone-", bytes, w, h, px);
    if (!buf) return false;

    int rga_fmt = toRgaFormat(source_format);
    rga_buffer_t src_buf = wrapbuffer_fd(source->fd, static_cast<int>(w), static_cast<int>(h),
                                          rga_fmt, static_cast<int>(source->w_stride),
                                          static_cast<int>(source->h_stride));
    rga_buffer_t dst_buf =
        wrapbuffer_fd(buf->fd, static_cast<int>(w), static_cast<int>(h), rga_fmt);

    im_rect rect = {0, 0, static_cast<int>(w), static_cast<int>(h)};
    IM_STATUS check_ret = imcheck(src_buf, dst_buf, rect, rect);
    if (check_ret != IM_STATUS_NOERROR) {
        std::cerr << "[RgaPreprocessor] cloneFrame imcheck başarısız: "
                  << imStrError(check_ret) << "\n";
        return false;
    }

    rga_buffer_t empty_pat{};
    im_rect empty_rect{};
    IM_STATUS proc_ret = improcess(src_buf, dst_buf, empty_pat, rect, rect, empty_rect, 0);
    if (proc_ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RgaPreprocessor] cloneFrame improcess başarısız: "
                  << imStrError(proc_ret) << "\n";
        return false;
    }

    // Çağıran (drawTrackedObjects) bu tampon üzerine RGA ile çizecek; CPU
    // ilk kez RtspStreamer::pushFrame'in memcpy'ında dokunuyor — invalidate'i
    // burada bir kez işaretlemek yeterli (bkz. referans aynı notu).
    impl_->pool.syncCpuReadBegin(buf);

    out_buffer = buf;
    return true;
}
