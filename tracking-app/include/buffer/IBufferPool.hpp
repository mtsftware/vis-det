#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <string>

// 04-IBufferPool.md sözleşmesinin soyut arayüzü. Sistemdeki TEK gerçek
// bellek sahibi — diğer tüm modüller ödünç alır, shared_ptr referans
// sayısı sıfıra inince buffer otomatik havuza döner (custom deleter).
//
// M6 fazında canlıda kanıtlanan düzeltme (CLAUDE.md "Canlıda Doğrulanmış
// Kurallar" §3): 04-IBufferPool.md'nin "her zaman uncached" kuralı yalnızca
// donanımdan-donanıma (RGA -> NPU fd) zero-copy yolu için doğrudur.
// Tüketici CPU ise (klasik-kopya yolu: rknn_inputs_set'in okuduğu RGA
// çıktıları) CACHED heap + DMA_BUF_IOCTL_SYNC zorunludur — uncached
// heap'ten kare başına 243KB CPU okuması FPS'i öldürüyordu. Bu yüzden
// heap türü havuz-başına seçilir, sabit değildir.
enum class HeapKind {
    CachedDma32,    // /dev/dma_heap/system-dma32 — tüketici CPU ise bunu seç
    UncachedDma32,  // /dev/dma_heap/system-uncached-dma32 — hw->hw zero-copy yolu
};

class IBufferPool {
public:
    virtual ~IBufferPool() = default;

    // Adlandırılmış, sabit boyutlu, sabit slot sayılı bir havuz yaratır.
    // Tüm slotlar burada, BİR KEZ tahsis edilir (kare başına tahsis yasak
    // — M6 kural #4). Aynı adla ve aynı parametrelerle tekrar çağrılırsa
    // true döner (idempotent); aynı ad farklı parametrelerle çağrılırsa
    // false (sessiz yeniden boyutlandırma yok — bu bir programlama hatasıdır).
    virtual bool createPool(const std::string& name, uint32_t buffer_size,
                            uint32_t slot_count, HeapKind heap) = 0;

    // Boş slot yoksa BLOKLAR (kasıtlı geri-basınç / backpressure, 04 §8 —
    // sınırsız kuyruklanmayı önler). shutdown() çağrılmışsa nullptr döner.
    // Dönen DmaBuffer'ın fd/virt_addr/size alanları havuz tarafından
    // doldurulur; width/height/stride/format/dtype ÜRETİCİNİN işidir
    // (havuz piksel anlamını bilmez — 04 §9).
    virtual DmaBufferPtr acquire(const std::string& name) = 0;

    // acquire'ın zaman aşımlı hali: timeout_ms içinde slot boşalmazsa
    // nullptr. Üretici yollarında (preprocess) deadlock'a karşı tercih
    // edilen varyant — sonsuz bloklanma yerine loglanabilir bir hata.
    virtual DmaBufferPtr tryAcquireFor(const std::string& name, int timeout_ms) = 0;

    // Tüm dolaşımdaki buffer'ların havuza dönmesini bekler, sonra tüm
    // fd/mmap'leri kapatır. release() public DEĞİL (04 §8) — iade yalnızca
    // shared_ptr deleter'ı üzerinden olur, kullanıcı kodu asla elle iade etmez.
    virtual void shutdown() = 0;
};
