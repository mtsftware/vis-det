#pragma once

#include "buffer/IBufferPool.hpp"

#include <cstdint>
#include <memory>
#include <string>

// DmaBufferPool: dma-heap tabanlı somut buffer havuzu implementasyonu.
// MPP/RGA zero-copy pipeline icin kullanilir.
class DmaBufferPool : public IBufferPool {
public:
    DmaBufferPool();
    ~DmaBufferPool() override;

    DmaBufferPool(const DmaBufferPool&) = delete;
    DmaBufferPool& operator=(const DmaBufferPool&) = delete;

    bool createPool(const std::string& name, uint32_t buffer_size,
                    uint32_t slot_count, HeapKind heap) override;
    DmaBufferPtr acquire(const std::string& name) override;
    DmaBufferPtr tryAcquireFor(const std::string& name, int timeout_ms) override;
    void shutdown() override;

    // Havuz yoksa olustur, varsa var olanı dondur (idempotent).
    bool ensurePool(const std::string& name, uint32_t buffer_size,
                    uint32_t slot_count, HeapKind heap);

    // Cached havuzdan CPU okuma yapilacaksa before çağrılır (DMA cache invalidate).
    void syncCpuReadBegin(const DmaBufferPtr& buffer);

    struct PoolStats {
        uint32_t slot_count = 0;
        uint32_t in_use = 0;
    };
    bool getStats(const std::string& name, PoolStats& out) const;
    
    struct Impl;
private:
    std::shared_ptr<Impl> impl_;
};