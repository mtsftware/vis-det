#pragma once

#include <cstdint>
#include <functional>
#include <memory>

// DmaBuffer: MPP decoder'dan gelen NV12 YUV frame için zero-copy buffer tanımı.
// MPP frame, mpp_frame_get_buffer() ile MppBuffer'a dönüştürülür ve bu
// struct içinde fd + virt_addr olarak capture edilir. release_fn, RAII ile
// mpp_frame_deinit() çağrısını garanti eder.
enum class PixelFormat { NV12, NV16, RGB888, BGR888, Unknown };
enum class DataType { UINT8, FP16, Unknown };

struct DmaBuffer {
    int fd = -1;
    void* virt_addr = nullptr;
    uint32_t size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t w_stride = 0;
    uint32_t h_stride = 0;
    PixelFormat format = PixelFormat::Unknown;
    DataType dtype = DataType::UINT8;
    uint64_t pts_ns = 0;

    // Kaynağa özel serbest bırakma işlemi (MPP: mpp_frame_deinit).
    std::function<void()> release_fn;

    ~DmaBuffer() {
        if (release_fn) {
            release_fn();
        }
    }
};

using DmaBufferPtr = std::shared_ptr<DmaBuffer>;