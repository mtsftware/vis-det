#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>
#include <string>

class RtspStreamer {
public:
    RtspStreamer(uint16_t port, std::string mount_path, uint32_t width,
                 uint32_t height, uint32_t fps = 30);
    ~RtspStreamer();

    RtspStreamer(const RtspStreamer&) = delete;
    RtspStreamer& operator=(const RtspStreamer&) = delete;

    bool start();
    void stop();
    
    // Boyutlari caltisma zamaninda guncelle (dynamic resize icin)
    void setDimensions(uint32_t width, uint32_t height);

    // pushFrame(): NV16 ise NV12'ye downsample, zaten NV12 ise direkt push
    // (color.md §5a ile uyumlu — mpph264enc her zaman NV12 bekler)
    void pushFrame(const DmaBufferPtr& frame);

    // NV12 formatını push eder — kaynak frame'den stride (w_stride) bilgisi
    // kullanılarak row-by-row tightly-packed kopyalama yapılır.
    void pushRawNV12(const DmaBufferPtr& frame);

    // NV16 formatını push eder — aynı row-by-row stratejisi.
    void pushRawNV16(const DmaBufferPtr& frame);

    // RGB888 formatını push eder — tightly-packed kopyalama.
    void pushRawRGB888(const DmaBufferPtr& frame);

    /* Callback'in Impl'e ulasabilmesi icin PUBLIC olmalidir */
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};