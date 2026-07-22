#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>
#include <string>

// KRİTİK NOT: Bu sınıf SADECE M4 doğrulaması içindir (decode pikselleri
// doğru mu, görsel olarak kanıtlamak). Gerçek zamanlı tracking pipeline'ının
// kritik yolunda YER ALMAZ. mpph264enc (GStreamer elementi, donanımsal ama
// GStreamer soyutlaması üzerinden) ve GstRTSPServer bilinçli olarak
// kullanılıyor — native MPP encoder API'si (rk_venc_cfg.h) burada DEĞİL,
// çünkü encoder config key-string'leri bu SDK sürümünde henüz doğrulanmadı
// (bkz. 06-system-architecture.md §9 "Açık Kalan Sorular").
class RtspStreamer {
public:
    RtspStreamer(uint16_t port, std::string mount_path, uint32_t width,
                 uint32_t height, uint32_t fps = 30);
    ~RtspStreamer();

    RtspStreamer(const RtspStreamer&) = delete;
    RtspStreamer& operator=(const RtspStreamer&) = delete;

    bool start();
    void stop();

    // MppDecoder::FrameCallback ile doğrudan uyumlu imza. NV12 içeriği
    // GstBuffer'a KOPYALANIR (bir CPU kopyası) — bu adım kritik path
    // olmadığından kabul edilebilir (bkz. sınıf üstü not).
    void pushFrame(const DmaBufferPtr& frame);

    // pushFrame'den farkı: kaynağın ZATEN doğru NV12 (width*height*3/2,
    // stride==width/height) olduğunu varsayar, NV16->NV12 downsample
    // YAPMAZ. RGA çıktısı (RgaPreprocessor) gibi zaten doğru formatta
    // gelen kaynaklar için kullanılır (M3).
    void pushRawNV12(const DmaBufferPtr& frame);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};