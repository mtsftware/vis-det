# IBufferPool — Yazılı Arayüz Sözleşmesi

> Kod değildir. `dma_buf` tabanlı, önceden tahsis edilmiş, sabit boyutlu bellek havuzunun sözleşmesidir. Sistemdeki **tek gerçek bellek sahibidir** — diğer tüm modüller ondan ödünç alır, geri iade eder.

**Kaynaklar:** Rapor 3 (RGA2 4GB sınırı), Rapor 5 (Havuzlama Mimarisi — ana kaynak), Rapor 4 (size_with_stride hesabı).

---

## 1. Sorumluluk Sınırı

`IBufferPool`, adlandırılmış **profiller** (`PoolProfile`) altında sabit sayıda, önceden tahsis edilmiş `DmaBuffer` slotu tutar. Her frame'de `malloc`/`free` yerine, havuzdan ödünç alma / geri iade döngüsü çalışır.

**Bilir:** Her profilin slot boyutunu (`size_with_stride`), tahsis bölgesini (`/dev/dma_heap/system-uncached-dma32`), kaç slot tuttuğunu.

**Bilmez:** RGA'nın ya da NPU'nun buffer içeriğiyle ne yaptığını — sadece yaşam döngüsünü (kim alıyor, kim iade ediyor) yönetir.

---

## 2. Neden Tek Bir Heap: `system-uncached-dma32`

İki ayrı sorunu **aynı anda** çözer:

1. **RGA2'nin 32-bit IOMMU sınırı** (Rapor 3): RGA2, çıktı 4GB fiziksel adres üzerine denk gelirse `IM_STATUS_INVALID_PARAM` ile başarısız olur; sistem SWIOTLB bounce buffer'a düşerse (`system` heap kullanılırsa) zero-copy felsefesi tamamen yıkılır. `dma32` heap'i bu sınırı garantiler.
2. **Cache coherency** (Rapor 5 + gerçek header bulgusu): `rknn_api.h`'da bir "cache flush'ı devre dışı bırak" flag'i **yoktur** (bkz. `02-IInferenceEngine.md` §8). Yani cache tutarlılığını RKNN API seviyesinde çözemiyoruz. Çözüm: baştan **uncached** bellek tahsis etmek — donanımdan donanıma (RGA→NPU) akan veri hiç CPU cache'ine uğramadığı için sorun fiziksel olarak oluşmuyor.

**Kesin kural:** Havuzdaki hiçbir slot `system` (genel, cached) heap'inden tahsis edilmez. Her zaman `system-uncached-dma32`.

---

## 3. Profil Tablosu (Sistemdeki Tüm Havuzlar)

| Profil adı | Kim kullanır | Boyut kaynağı | Slot sayısı (öneri) |
|---|---|---|---|
| `DecodeFramePool` | MPP decode çıktısı (NV12) | Video çözünürlüğü × 1.5 (NV12) | MPP kendi `MppBufferGroup`'unu yönetir — bu, ayrı bir mekanizma, §7'ye bakın |
| `DetectorInputPool` | RGA→Detector girişi | Detector `ModelProfile.size_with_stride` | 2–3 (çift/üçlü tamponlama) |
| `TrackerTemplateInputPool` | RGA→template_backbone girişi | Template `ModelProfile.size_with_stride` | 1–2 (nadiren kullanılır) |
| `TrackerSearchInputPool` | RGA→search_backbone girişi | Search `ModelProfile.size_with_stride` | 2–3 |
| `DetectorOutputPool` / `TrackerOutputPool` | Çıktı tensörleri (opsiyonel, zero-copy çıktı isteniyorsa) | İlgili `ModelProfile` | 2 |

**Bağımlılık:** Her profil boyutu, ilgili `IInferenceEngine::queryProfile()` çağrısından **sonra** belirlenir. `IBufferPool::createPool()` bu yüzden pipeline başlatma sırasında en son adımlardan biridir (bkz. `06-system-architecture.md` §Başlatma Sırası).

---

## 4. Durum Makinesi (Android BufferQueue'dan Uyarlanmış)

```
DEQUEUED  →  QUEUED  →  ACQUIRED  →  FREE  →  (DEQUEUED)
```

| Durum | Anlamı | Kim tetikler |
|---|---|---|
| `DEQUEUED` | Havuzdan boş slot alındı, tamamen üreticiye ait | `IPreprocessor` (RGA yazmadan önce) |
| `QUEUED` | Üretici işini bitirdi, tüketici kuyruğuna bırakıldı | `IPreprocessor::process()` dönüşü |
| `ACQUIRED` | Tüketici (bir `IInferenceEngine` instance'ı) buffer'ı aldı, referans sayısı arttı | `IInferenceEngine::bindInput()` |
| `FREE` | Tüm tüketiciler işini bitirdi, referans sayısı sıfıra indi, havuza geri döndü | Otomatik — `shared_ptr` custom deleter |

---

## 5. Sahiplik Modeli: `shared_ptr<DmaBuffer>` + Custom Deleter (RAII)

Buffer, çıplak bir `int fd` olarak **değil**, her zaman `shared_ptr<DmaBuffer>` olarak dolaşır. Havuz, `shared_ptr` oluştururken bir custom deleter atar:

```
shared_ptr<DmaBuffer> buf(raw_ptr, [pool](DmaBuffer* b) {
    pool->release(b);   // havuza geri iade, "free" ETMEZ
});
```

Bu sayede:
- Bir buffer birden fazla tüketiciye (örn. hem tracker hem opsiyonel kayıt/encode thread'i) `shared_ptr` kopyası olarak dağıtılabilir; referans sayısı sıfıra inene kadar havuza dönmez.
- Kilitlenmesiz (`lock-free`), kullanıcı-alanı seviyesinde referans sayımı — kernel DMA kilidi gerekmez.
- Erken serbest bırakma ("use-after-free") ve sızıntı (asla geri dönmeyen buffer) riskleri ortadan kalkar.

---

## 6. `DmaBuffer` Veri Yapısı (Kanonik Tanım)

> Tam tanım `06-system-architecture.md` §Ortak Veri Tipleri'nde. Burada sadece `IBufferPool`'un doldurduğu alanlar listelenir:

```
fd              : int32_t     // dma_buf file descriptor
virt_addr       : void*       // mmap edilmiş sanal adres (debug/slow-path için)
size            : uint32_t    // size_with_stride
w_stride        : uint32_t
h_stride        : uint32_t
format          : PixelFormat // NV12 / RGB888 / vb.
dtype           : DataType    // UINT8 / FP16
pool_origin     : IBufferPool* // custom deleter'ın hangi havuza iade edeceğini bilmesi için
```

---

## 7. MPP Decode Havuzu — Ayrı Bir Mekanizma, Aynı Felsefe

MPP'nin kendi bellek yönetimi (`MppBufferGroup`) `IBufferPool`'un **dışındadır** ama aynı "sabit, önceden tahsisli, runtime'da malloc yok" ilkesine uyar:

- Mod: **Yarı Dahili (Semi-Internal)** — `mpp_buffer_group_get_internal(&grp, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)`.
- Sınırlama: `mpp_buffer_group_limit_config` ile maksimum kare sayısı sabitlenir (örn. 24).
- Referans sayımı: MPP'nin kendi dahili mekanizması (`mpp_frame_deinit`/`mpp_buffer_put`) — `IBufferPool`'un `shared_ptr` modeliyle **karıştırılmaz**, decode çıktısı `IPreprocessor`'a girerken bir `DmaBuffer` sarmalayıcısına (wrapper) dönüştürülür ve o andan itibaren `IBufferPool`'un kurallarına tabi olur.

---

## 8. Metod Sözleşmesi (pseudo-imza)

```
createPool(name: string, profile: ModelProfile, slot_count: uint32) -> void
    // system-uncached-dma32 heap'inden slot_count adet DMA_HEAP_IOCTL_ALLOC.

acquire(pool_name: string) -> shared_ptr<DmaBuffer>
    // DEQUEUED durumuna geçirir. Havuz boşsa çağıran thread bloklanır
    // (std::condition_variable) — bu kasıtlı bir "geri basınç" (backpressure)
    // mekanizmasıdır, sınırsız kuyruklanmayı önler.

release(buf: DmaBuffer*) -> void
    // Sadece custom deleter tarafından çağrılır. Kullanıcı kodu asla
    // doğrudan çağırmaz.

shutdown() -> void
    // Tüm fd'leri kapatır, dma_heap açık dosya tanımlayıcısını kapatır.
```

---

## 9. Açıkça Yapmadıkları

- RGA veya RKNN çağrısı yapmaz — saf bir tahsis/yaşam-döngüsü yöneticisidir.
- Buffer içeriğine dokunmaz (piksel/tensör verisiyle ilgilenmez).
- Cache flush/invalidate çağrısı yapmaz — çünkü uncached heap sayesinde buna hiç gerek yoktur.
