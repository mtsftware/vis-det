#include "preprocess/RgaPreprocessor.hpp"

#include "buffer/DmaBufferPool.hpp"

#include "im2d.h"
#include "rga.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

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

// M7 GEÇİŞİ: M6'daki geçici ring önbelleği (kalıcı heap fd + boyut-başına
// 4 slot + cached-heap/dma-buf-sync) buradan sökülüp DmaBufferPool'a
// (04-IBufferPool.md) taşındı. Bu sınıf artık 01-IPreprocessor.md §7'ye
// uygun: kendi buffer'ını TAHSİS ETMEZ, havuzdan ödünç alır; iade,
// shared_ptr referans sayısı sıfıra inince otomatik (custom deleter).
//
// Havuzlar burada ensurePool ile İLK KULLANIMDA kuruluyor (boyutlar ancak
// configure/crop çağrısında netleşiyor); orkestratör fazında bu isimler
// PipelineOrchestrator tarafından başlatma sırasında createPool ile
// önceden kurulmalı (04 §3). Heap türü: CachedDma32 — bu çıktıların
// tüketicisi CPU (rknn_inputs_set / PPM dump / streamer kopyası), bkz.
// CLAUDE.md "Canlıda Doğrulanmış Kurallar" §3.
//
// YAŞAM DÖNGÜSÜ (M6'daki "3 çağrı geçerli" ring sözleşmesinin YERİNİ ALDI):
// dönen DmaBuffer'lar gerçek referans sayımlı — çağıran istediği kadar
// tutabilir; ancak aynı havuzdan 4 buffer'ı aynı anda tutmak üreticiyi
// bloklar/zaman aşımına düşürür (kasıtlı geri basınç).
struct RgaPreprocessor::Impl {
    uint32_t target_width = 0;
    uint32_t target_height = 0;
    RgaPixelFormat target_format = RgaPixelFormat::NV12;
    bool configured = false;

    DmaBufferPool pool;

    // Havuzdan RGB888/hedef-format buffer alır; DmaBuffer meta alanlarını
    // üretici sorumluluğu gereği (04 §9) burada doldurur.
    DmaBufferPtr acquireBuffer(const char* tag, size_t bytes, uint32_t w, uint32_t h,
                                PixelFormat px_fmt) {
        std::string name = std::string(tag) + std::to_string(bytes);
        if (!pool.ensurePool(name, static_cast<uint32_t>(bytes), 4,
                              HeapKind::CachedDma32)) {
            std::cerr << "[RgaPreprocessor] havuz kurulamadı: " << name << "\n";
            return nullptr;
        }
        // Bloklamalı acquire DEĞİL: preprocess yolunda deadlock yerine
        // loglanabilir hata tercih edilir (tüketici buffer'ları sızdırıyorsa
        // bunu sessiz kilitlenme değil, log olarak görmek isteriz).
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
    impl_->target_height = target_height;  // yükseklik hizası genelde
                                            // sorun çıkarmıyor (Rapor 3)
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

    // Letterbox oranı: aspect-ratio korunarak hedefe sığdırma (Rapor 3).
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

    // ADIM 1: Zemini gri ile doldur (immakeBorder DEĞİL — bkz. header notu).
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

    // CPU okuyacaksa (streamer pushRawNV12 vb.) cache invalidate gerekli.
    impl_->pool.syncCpuReadBegin(buf);

    out_buffer = buf;
    return true;
}

bool RgaPreprocessor::cropTargetCentric(const DmaBufferPtr& source,
                                         RgaPixelFormat source_format, float center_x,
                                         float center_y, int context_size,
                                         int model_size, uint8_t pad_r, uint8_t pad_g,
                                         uint8_t pad_b, DmaBufferPtr& out_buffer) {
    if (!source || source->fd < 0 || context_size <= 0 || model_size <= 0) {
        return false;
    }

    // get_subwindow_tracking() (lighttrack.cpp) ile AYNI matematik —
    // sadece piksel islemleri (pad+crop+resize) burada RGA'ya devrediliyor.
    float c = static_cast<float>(context_size + 1) / 2.0f;
    int orig_xmin = static_cast<int>(center_x - c + 0.5f);
    int orig_xmax = orig_xmin + context_size - 1;
    int orig_ymin = static_cast<int>(center_y - c + 0.5f);
    int orig_ymax = orig_ymin + context_size - 1;

    int src_w = static_cast<int>(source->width);
    int src_h = static_cast<int>(source->height);

    int left_pad = std::max(0, -orig_xmin);
    int top_pad = std::max(0, -orig_ymin);
    int right_pad = std::max(0, orig_xmax - src_w + 1);
    int bottom_pad = std::max(0, orig_ymax - src_h + 1);

    int visible_x0 = orig_xmin + left_pad;
    int visible_y0 = orig_ymin + top_pad;
    int visible_x1 = orig_xmax + 1 - right_pad;
    int visible_y1 = orig_ymax + 1 - bottom_pad;

    // RGA kural #2 (CLAUDE.md canlı kurallar): YUV kaynak crop rect'i 2'ye
    // hizalı olmak zorunda — hizasızsa imcheck sessizce reddeder, crop boş
    // (pad rengi) kalır ama fonksiyon başarılı görünürdü (M6'daki anlamsız
    // takibin kök nedeni). Kaynak çözünürlükler her zaman çift olduğundan
    // aşağı/yukarı 2'ye yuvarlamak sınır içinde kalır.
    visible_x0 -= (visible_x0 & 1);
    visible_y0 -= (visible_y0 & 1);
    if (visible_x1 & 1) visible_x1 = std::min(visible_x1 + 1, src_w);
    if (visible_y1 & 1) visible_y1 = std::min(visible_y1 + 1, src_h);

    left_pad = std::max(0, visible_x0 - orig_xmin);
    top_pad = std::max(0, visible_y0 - orig_ymin);
    right_pad = std::max(0, (orig_xmax + 1) - visible_x1);
    bottom_pad = std::max(0, (orig_ymax + 1) - visible_y1);

    double ratio = static_cast<double>(model_size) / context_size;

    // RGA kural #2 devamı: RGB888 hedef stride 16-hizalı olmak zorunda.
    // model_size 16'ya bölünmüyorsa (exemplar=127) 16-hizalı scratch'e
    // yazdırıp CPU'da sıkı-paketli buffer'a repack ediyoruz; bölünüyorsa
    // (instance=288) scratch doğrudan sonuçtur (fast path).
    int rga_stride = ((model_size + 15) / 16) * 16;
    bool needs_repack = (rga_stride != model_size);

    size_t rga_size = formatByteSize(RgaPixelFormat::RGB888, rga_stride, rga_stride);
    DmaBufferPtr scratch =
        impl_->acquireBuffer("crop-scratch-", rga_size, static_cast<uint32_t>(model_size),
                              static_cast<uint32_t>(model_size), PixelFormat::RGB888);
    if (!scratch) return false;

    rga_buffer_t dst_buf = wrapbuffer_fd(scratch->fd, model_size, model_size,
                                          toRgaFormat(RgaPixelFormat::RGB888), rga_stride,
                                          rga_stride);

    im_rect full_dst_rect = {0, 0, model_size, model_size};
    int pad_color = (static_cast<int>(pad_r) << 16) | (static_cast<int>(pad_g) << 8) |
                     static_cast<int>(pad_b);
    imfill(dst_buf, full_dst_rect, pad_color);

    bool has_visible_region = (visible_x1 > visible_x0) && (visible_y1 > visible_y0);

    if (has_visible_region) {
        int src_rga_fmt = toRgaFormat(source_format);
        rga_buffer_t src_buf = wrapbuffer_fd(
            source->fd, static_cast<int>(source->width),
            static_cast<int>(source->height), src_rga_fmt,
            static_cast<int>(source->w_stride), static_cast<int>(source->h_stride));

        im_rect src_rect = {visible_x0, visible_y0, visible_x1 - visible_x0,
                             visible_y1 - visible_y0};

        int dst_x0 = static_cast<int>(std::round(left_pad * ratio));
        int dst_y0 = static_cast<int>(std::round(top_pad * ratio));
        int dst_x1 = static_cast<int>(
            std::round((left_pad + (visible_x1 - visible_x0)) * ratio));
        int dst_y1 = static_cast<int>(
            std::round((top_pad + (visible_y1 - visible_y0)) * ratio));

        // RGA kural #2 devamı: hedef alt-dikdörtgen kanvasın tam kenarına
        // dokunamaz VE 2px'ten küçük olamaz (her ikisi de canlı testte
        // imcheck reddiyle kanıtlandı) — en fazla 1px'lik görünmez kırpma
        // payıyla clamp'liyoruz.
        dst_x0 = std::max(0, std::min(dst_x0, model_size - 3));
        dst_y0 = std::max(0, std::min(dst_y0, model_size - 3));
        dst_x1 = std::max(dst_x0 + 2, std::min(dst_x1, model_size - 1));
        dst_y1 = std::max(dst_y0 + 2, std::min(dst_y1, model_size - 1));

        // RGA kural #2 devamı-2 (M7 regresyon koşusunda teşhis edildi):
        // kaynak YUV olduğunda RGA, HEDEF dikdörtgende de 2-hizalama
        // istiyor — hedef RGB888 olsa bile. Reddedilen rect'lerin ortak
        // özelliği tek sayılı offset'ti (x=1 ya da y=1: hedef kare kenara
        // yakınken pad payının clamp'i 1px offset üretiyor); iç bölge
        // crop'ları [0,0,...] hep geçiyordu. imStrError'un "smaller than
        // 2 pixels" mesajı yanıltıcı — gerçek kısıt hizalama. Offset'i
        // aşağı, boyutu aşağı 2'ye yuvarlıyoruz (en fazla 1px kayma,
        // görsel olarak önemsiz; crop'un BOŞ kalmasından her zaman iyi).
        dst_x0 &= ~1;
        dst_y0 &= ~1;
        int dst_w = std::max(2, (dst_x1 - dst_x0) & ~1);
        int dst_h = std::max(2, (dst_y1 - dst_y0) & ~1);
        if (dst_x0 + dst_w > model_size - 1) dst_w -= 2;
        if (dst_y0 + dst_h > model_size - 1) dst_h -= 2;

        im_rect dst_rect = {dst_x0, dst_y0, dst_w, dst_h};

        rga_buffer_t empty_pat{};
        im_rect empty_rect{};

        IM_STATUS check_ret = imcheck(src_buf, dst_buf, src_rect, dst_rect);
        if (check_ret == IM_STATUS_NOERROR) {
            IM_STATUS proc_ret =
                improcess(src_buf, dst_buf, empty_pat, src_rect, dst_rect, empty_rect, 0);
            if (proc_ret != IM_STATUS_SUCCESS) {
                std::cerr << "[RgaPreprocessor] cropTargetCentric improcess başarısız: "
                          << imStrError(proc_ret) << "\n";
            }
        } else {
            std::cerr << "[RgaPreprocessor] cropTargetCentric imcheck başarısız: "
                      << imStrError(check_ret) << "\n";
        }
    }
    // NOT: has_visible_region false ise (hedef tamamen kare dışında),
    // dst tamamen pad rengiyle dolu kalır — improcess hiç çağrılmaz.

    if (!needs_repack) {
        // Fast path (her karedeki search crop, 288): scratch zaten
        // sıkı-paketli. CPU (rknn memcpy) okumadan önce cache invalidate.
        impl_->pool.syncCpuReadBegin(scratch);
        out_buffer = scratch;
        return true;
    }

    // model_size 16-hizalı değil (exemplar=127): RGA'nın stride'lı yazdığı
    // scratch'i sıkı-paketli buffer'a CPU satır kopyasıyla dönüştür.
    // Scratch device tarafından yazıldı -> okumadan önce invalidate.
    // Tight buffer'a SADECE CPU dokunuyor (yaz + rknn memcpy oku) -> sync
    // gerekmez. Scratch bu fonksiyonun sonunda scope'tan çıkınca havuza
    // otomatik döner (RAII — M6 ring'inin elle yönetimi yok artık).
    impl_->pool.syncCpuReadBegin(scratch);

    size_t tight_size = formatByteSize(RgaPixelFormat::RGB888, model_size, model_size);
    DmaBufferPtr tight =
        impl_->acquireBuffer("crop-tight-", tight_size, static_cast<uint32_t>(model_size),
                              static_cast<uint32_t>(model_size), PixelFormat::RGB888);
    if (!tight || !tight->virt_addr || !scratch->virt_addr) return false;

    const uint8_t* src_bytes = static_cast<const uint8_t*>(scratch->virt_addr);
    uint8_t* dst_bytes = static_cast<uint8_t*>(tight->virt_addr);
    size_t src_row_bytes = static_cast<size_t>(rga_stride) * 3;
    size_t dst_row_bytes = static_cast<size_t>(model_size) * 3;
    for (int y = 0; y < model_size; ++y) {
        std::memcpy(dst_bytes + static_cast<size_t>(y) * dst_row_bytes,
                    src_bytes + static_cast<size_t>(y) * src_row_bytes, dst_row_bytes);
    }

    out_buffer = tight;
    return true;
}

bool RgaPreprocessor::cloneFrame(const DmaBufferPtr& source, RgaPixelFormat source_format,
                                   DmaBufferPtr& out_buffer) {
    if (!source || source->fd < 0) return false;

    uint32_t w = source->width;
    uint32_t h = source->height;

    // Hedef her zaman SIKI-PAKETLİ (w_stride=w, h_stride=h) — acquireBuffer
    // zaten böyle dolduruyor. Kaynağın kendi stride'ı (dolgu payı olabilir)
    // src_buf'a ayrıca veriliyor; RGA farklı stride'lar arasında kopyayı
    // sorunsuz hallediyor (crop yolundaki improcess ile aynı prensip).
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

    // Kopyadan SONRA çağıran (drawBboxOutline) bu tampon üzerine YİNE RGA
    // ile çizecek — asıl CPU okuması RtspStreamer::pushFrame'in memcpy'ı,
    // o yüzden invalidate'i çağıran, çizimden SONRA kendi akışında
    // (main_m8/m9_test.cpp) tekrar tetikleyecek şekilde bırakmak yerine
    // burada bir kez işaretlemek yeterli: sonraki RGA yazımı (çizim) zaten
    // aynı cache durumunu koruyor, CPU ilk kez pushFrame'de dokunuyor.
    impl_->pool.syncCpuReadBegin(buf);

    out_buffer = buf;
    return true;
}

bool RgaPreprocessor::flipHorizontal(const DmaBufferPtr& source, DmaBufferPtr& out_buffer) {
    if (!source || source->fd < 0) return false;

    size_t dst_size = formatByteSize(RgaPixelFormat::RGB888, source->width, source->height);
    DmaBufferPtr buf = impl_->acquireBuffer("flip-", dst_size, source->width,
                                             source->height, PixelFormat::RGB888);
    if (!buf) return false;

    rga_buffer_t src_buf = wrapbuffer_fd(
        source->fd, static_cast<int>(source->width), static_cast<int>(source->height),
        toRgaFormat(RgaPixelFormat::RGB888), static_cast<int>(source->w_stride),
        static_cast<int>(source->h_stride));
    rga_buffer_t dst_buf =
        wrapbuffer_fd(buf->fd, static_cast<int>(source->width),
                       static_cast<int>(source->height), toRgaFormat(RgaPixelFormat::RGB888));

    IM_STATUS ret = imflip(src_buf, dst_buf, IM_HAL_TRANSFORM_FLIP_H);
    if (ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RgaPreprocessor] imflip başarısız: " << imStrError(ret) << "\n";
        return false;
    }

    impl_->pool.syncCpuReadBegin(buf);
    out_buffer = buf;
    return true;
}