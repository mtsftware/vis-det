#pragma once

#include "buffer/IBufferPool.hpp"

#include <cstdint>
#include <memory>
#include <string>

// IBufferPool'un dma-heap tabanlı somut implementasyonu (04-IBufferPool.md).
// M6 fazında RgaPreprocessor'ın içinde kanıtlanan ring-önbellek mantığının
// (kalıcı heap fd, önceden tahsis, cached-heap + dma-buf sync) sözleşmeye
// oturtulmuş halidir — o geçici ring artık bu sınıfa taşındı.
//
// Sahiplik/yaşam süresi: acquire() gerçek referans sayımlı DmaBufferPtr
// döner; son referans düştüğünde custom release zinciri slotu havuza iade
// eder. Deleter, havuz nesnesine weak_ptr üzerinden bağlanır — havuz
// yok edildikten SONRA serbest kalan bir buffer çökme yaratmaz (iade
// sessizce no-op olur); yine de doğru kullanım havuzu tüketicilerden
// uzun yaşatmaktır (shutdown zaten dolaşımdaki buffer'ları bekler).
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

    // createPool'un "varsa dokunma, yoksa yarat" hali. Orkestratör çağı
    // geldiğinde havuzlar başlatma sırasında createPool ile açıkça
    // kurulmalı (04 §3 bağımlılık notu); bu yardımcı, boyutların ancak
    // ilk kullanımda netleştiği geçiş dönemi (RgaPreprocessor iç havuzu)
    // için pragmatik bir kolaylıktır.
    bool ensurePool(const std::string& name, uint32_t buffer_size,
                    uint32_t slot_count, HeapKind heap);

    // CACHED havuzdan gelen ve az önce DONANIM (RGA) tarafından yazılmış
    // bir buffer'ı CPU okumadan önce çağrılır — cache invalidate
    // (DMA_BUF_IOCTL_SYNC START|READ). Uncached havuzlarda no-op. Açılan
    // READ bracket'ı, buffer havuza iade edilirken otomatik kapatılır.
    // (SADECE cihazın yazıp CPU'nun okuduğu yol için gerekli; CPU'nun
    // yazıp CPU'nun okuduğu buffer'larda çağrılmaz.)
    void syncCpuReadBegin(const DmaBufferPtr& buffer);

    struct PoolStats {
        uint32_t slot_count = 0;
        uint32_t in_use = 0;
    };
    bool getStats(const std::string& name, PoolStats& out) const;

    struct Impl;

private:
    // shared_ptr: dolaşımdaki buffer'ların deleter'ları Impl'e weak_ptr ile
    // bağlanır (bkz. sınıf üstü yaşam süresi notu).
    std::shared_ptr<Impl> impl_;
};
