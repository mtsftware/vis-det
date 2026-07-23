#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <string>

// IBufferPool: DMA heap tabanlı buffer havuzu soyu arayuzu.
enum class HeapKind {
    CachedDma32,    // /dev/dma_heap/system-dma32 — tuketici CPU ise
    UncachedDma32,  // /dev/dma_heap/system-uncached-dma32 — hw->hw zero-copy
};

class IBufferPool {
public:
    virtual ~IBufferPool() = default;

    virtual bool createPool(const std::string& name, uint32_t buffer_size,
                            uint32_t slot_count, HeapKind heap) = 0;

    virtual DmaBufferPtr acquire(const std::string& name) = 0;

    virtual DmaBufferPtr tryAcquireFor(const std::string& name, int timeout_ms) = 0;

    virtual void shutdown() = 0;
};