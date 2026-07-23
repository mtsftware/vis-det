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

    void pushRawNV12(const DmaBufferPtr& frame);

    /* Callback'in Impl'e ulasabilmesi icin PUBLIC olmalidir */
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};
