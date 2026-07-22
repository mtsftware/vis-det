#pragma once

#include <cstdint>
#include <functional>
#include <memory>

// 06-system-architecture.md §3'teki kanonik DmaBuffer tanımının somut
// implementasyonu. Farklı kaynaklar (MPP decode, RGA çıktısı, dma-heap
// havuzu) farklı serbest bırakma mekanizmalarına sahip olduğundan,
// bu mekanizma nesnenin İÇİNE (release_fn) gömülür — böylece tüm
// tüketiciler tek tip bir shared_ptr<DmaBuffer> görür, kaynağı bilmesi
// gerekmez.
enum class PixelFormat { NV12, NV16, RGB888, BGR888, Unknown };
enum class DataType { UINT8, FP16, Unknown };

struct DmaBuffer {
    int fd = -1;
    void* virt_addr = nullptr;  // debug / slow-path için, fast path'te kullanılmaz
    uint32_t size = 0;          // size_with_stride
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t w_stride = 0;
    uint32_t h_stride = 0;
    PixelFormat format = PixelFormat::Unknown;
    DataType dtype = DataType::UINT8;
    uint64_t pts_ns = 0;

    // Kaynağa özel serbest bırakma işlemi (MPP: mpp_frame_deinit,
    // dma-heap havuzu: pool->release, vb.). shared_ptr referans sayısı
    // sıfıra indiğinde bu nesnenin destructor'ı çağrılır, o da
    // release_fn'i tetikler — RAII zinciri burada kapanır.
    std::function<void()> release_fn;

    ~DmaBuffer() {
        if (release_fn) {
            release_fn();
        }
    }
};

using DmaBufferPtr = std::shared_ptr<DmaBuffer>;