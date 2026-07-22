# IPreprocessor — Yazılı Arayüz Sözleşmesi

> Kod değildir. Bu belge, `IPreprocessor` soyutlamasının sorumluluk sınırlarını, çağrı sırasını ve davranış kurallarını tanımlar. İmplementasyon (`RgaPreprocessor`) bu sözleşmeye uymak zorundadır.

**Kaynaklar:** Rapor 1 (Hibrit Mimari), Rapor 2 (rknpu2 Stride/Alignment), Rapor 3 (RGA Letterbox), Rapor 5 (Buffer Havuzlama), gerçek `rknn_api.h`.

---

## 1. Sorumluluk Sınırı

`IPreprocessor`'ın tek görevi: bir `DmaBuffer` (ham decode çıktısı, NV12) alıp, RGA donanımını kullanarak NPU'nun beklediği formatta **başka bir `DmaBuffer`** üretmektir. CPU hiçbir zaman piksel verisine dokunmaz (fast path'te).

**Bilir:** Kaynak buffer'ın (fd, width, height, stride, format=NV12), hedef `ModelProfile`'ın (bkz. `06-system-architecture.md`) stride/format/dtype gereksinimlerini, hangi RGA çekirdeğinin (RGA2/RGA3) hangi işlemi desteklediğini.

**Bilmez:** NPU'nun nasıl çalıştığını, model çıktısının nasıl yorumlanacağını, tracker/detector state machine'ini, hangi thread'de çalıştığını (orkestratör tarafından çağrılır, kendi thread'ini yönetmez).

---

## 2. Çağrı Sırası — Kritik Bağımlılık

`IPreprocessor` **asla önce konfigüre edilemez**. Sıra kesinlikle şudur:

```
1. İlgili IInferenceEngine::initialize() çağrılır
2. IInferenceEngine::queryProfile() ile ModelProfile alınır
   (w_stride, h_stride, size_with_stride, dtype, target width/height)
3. IPreprocessor::configure(ModelProfile) çağrılır — RGA hedef parametreleri
   BU NOKTADA sabitlenir
4. IBufferPool bu ModelProfile.size_with_stride'a göre havuz oluşturur
5. Ancak bundan sonra IPreprocessor::process() çağrılabilir
```

Bu sıra ihlal edilirse (örn. RGA'ya sabit 640×640 varsayımıyla önceden konfigüre edilip NPU'ya "uydurulmaya" çalışılırsa), Rapor 2'nin tarif ettiği **sessiz veri bozulması** ("Çiçek Ekran") riski doğar — `RKNN_SUCC` döner ama çıktı çöptür. Bu, test sırasında yakalanamayacak bir hata sınıfıdır; bu yüzden sıra bir *öneri* değil, **zorunluluktur**.

---

## 3. İki Farklı Çalışma Modu

`IPreprocessor` tek bir davranış değil, **iki farklı geometrik strateji** uygular. Hangisinin kullanılacağı, `configure()`'a verilen `PreprocessMode` ile belirlenir:

### 3.1 `PreprocessMode::FullFrameLetterbox` (Detector için)

Tüm kareyi, en-boy oranını koruyarak hedef kare boyutuna (örn. 640×640) sığdırır.

**RGA çağrı sırası (kesin, Rapor 3'ten):**
1. `imfill(hedef_buffer, {0,0,W,H}, gray_114)` — zemini nötr griyle doldur
2. `improcess(kaynak_NV12, hedef_buffer, srect=tam_kare, drect=merkez_hesaplı_dikdörtgen)` — donanımsal resize+format dönüşümü, merkeze yerleştir

**KESİNLİKLE YASAK:** `immakeBorder()` fonksiyonu. Gerekçe: renk doldurma işlemi zorunlu olarak RGA2 çekirdeğine düşer, RGA2'nin 32-bit IOMMU sınırı (4GB) çoğu standart Linux dağıtımında `system-uncached-dma32` heap'i kapalı olduğundan üretim ortamında `IM_STATUS_INVALID_PARAM` ile çöker (Rapor 3, Issue #143).

### 3.2 `PreprocessMode::TargetCentricCrop` (Tracker için — template & search)

> **Not:** Bu mod 6 araştırma raporunun hiçbirinde doğrudan ele alınmadı; Siamese-tarzı takipçilerin (NanoTrack/LightTrack) standart ön işleme mantığından, aynı RGA zero-copy temeline oturtularak sentezlenmiştir.

Tüm kareyi değil, **hedefin son bilinen konumu etrafında** bir kare bölge kırpar ve sabit boyuta ölçekler. İki alt-profil vardır:

| Profil | Ne zaman kullanılır | Bağlam çarpanı (context factor) |
|---|---|---|
| `TemplateProfile` | Sadece `ITracker::initialize()` içinde | Dar (hedefi sıkı çevreler) |
| `SearchProfile` | Her karede | Geniş (hedefin kaçabileceği alanı kapsar) |

**RGA çağrı sırası:**
1. Kırpılacak kaynak dikdörtgeni hesapla: `crop_size = target_size × context_factor`, merkez = son bilinen bbox merkezi
2. Kırpma kare sınırlarını taşıyorsa (görüntü kenarına yakın hedef), `imfill()` ile önce hedef buffer ortalama-renk veya kenar-rengiyle doldurulur, sonra `improcess(srect=kırpılmış_ve_kenara_kırpılmış_alan, drect=uygun_offset)` ile yerleştirilir. Bu, `FullFrameLetterbox`'daki iki-adımlı stratejiyle **aynı donanımsal mantığı** kullanır (RGA hedef dikdörtgen dışına asla dokunmaz), sadece kırpma merkezi sabit değil, dinamiktir.
3. `improcess` ile `crop_size → target_size` (örn. 255×255 → 255×255 search, ölçek 1:1 olabilir, ya da modele göre farklı) donanımsal resize.

**Zorunlu ek çıktı — Ters Dönüşüm Bilgisi:**
Bu modda `process()`, `DmaBuffer`'a ek olarak bir `CropTransform{origin_x, origin_y, scale}` yapısı döndürmek **zorundadır**. `IPostprocessor`, tracker'ın ürettiği bbox'ı bu bilgiyle tam kare koordinatına geri çevirir. Bu bilgi olmadan `IPostprocessor` çalışamaz — bu, `IPreprocessor` ↔ `IPostprocessor` arasındaki tek doğrudan veri bağımlılığıdır.

---

## 4. Stride ve Hizalama Kuralları (Kesin, Rapor 2 & 3)

- Genişlik formülü, **her zaman**: `width_px = ((target_w + 15) / 16) * 16`
- RGA3 üzerinden RGB_888/BGR_888 çıktılarda width stride 16-byte hizalı olmak **zorunda**.
- `configure()` sırasında alınan `ModelProfile.w_stride` değeri RGA'nın `wstride` parametresine **birebir** yazılır — `IPreprocessor` kendi stride hesabını NPU'nunkinin üzerine yazmaz, NPU'nunkini esas alır.

---

## 5. Veri Tipi (dtype) Farkındalığı — Kritik Güvenlik Kuralı

`ModelProfile.precision` alanına göre:

| Precision | RGA çıktısı | Sonraki adım |
|---|---|---|
| `INT8` / `UINT8` | RGA doğrudan UINT8 RGB üretir | Doğrudan NPU'ya, `pass_through` NPU tarafında yönetilir |
| `FP16` | RGA UINT8 üretir | **CPU'ya asla gitmeden**, RGA veya Mali-G610 GPU (OpenCL) ile FP16'ya dönüştürülmesi gerekir — bu, `IPreprocessor`'ın **üçüncü bir adımı** |

**MUTLAK YASAK:** FP16 hedefli bir `ModelProfile` için, ham UINT8 RGB verisini hiçbir dönüşüm yapmadan NPU'ya vermek. Bu, Rapor 4'ün tarif ettiği "bayt yozlaşması" hatasına yol açar — segfault vermez, `RKNN_SUCC` döner, ama NPU iki bitişik UINT8 baytını tek bir anlamsız FP16 sayısı olarak yorumlar. `IPreprocessor`, `ModelProfile.precision == FP16` gördüğünde bu üçüncü dönüşüm adımını **otomatik olarak** zincire eklemek zorundadır; bu, çağıranın (orkestratör/`ITracker`) karar vereceği bir şey değildir.

---

## 6. Fast Path / Slow Path Ayrımı

`configure()` sırasında RGA'nın üretebileceği stride ile `ModelProfile.w_stride` karşılaştırılır:

- **Eşleşiyorsa (fast path):** RGA çıktısı doğrudan NPU'nun `dma_buf` fd'sine yazılır. Sıfır kopya.
- **Eşleşmiyorsa (slow path):** RGA yine kullanılır ama sonuç ara bir buffer'a yazılır, ardından CPU'da satır-satır (`memcpy` with stride) doğru stride'a kopyalanır. Performans kaybı kabul edilir, doğruluk korunur.

`IPreprocessor`, hangi path'in aktif olduğunu **her frame için** metrik/tracking katmanına raporlamalıdır (bkz. `06-system-architecture.md` § İzleme). Bu bilgi olmadan bir performans regresyonu teşhis edilemez.

---

## 7. Buffer Yaşam Döngüsü İlişkisi

`IPreprocessor` kendi buffer'ını tahsis etmez — `IBufferPool`'dan `DEQUEUED` durumunda bir slot alır (bkz. `04-IBufferPool.md`), üzerine yazar, `QUEUED` durumuna geçirip döndürür. `IPreprocessor`, `FREE`/geri-iade mantığını bilmez; bu `shared_ptr<DmaBuffer>`'ın custom deleter'ının işidir.

---

## 8. Metod Sözleşmesi (pseudo-imza, kod değil)

```
configure(profile: ModelProfile, mode: PreprocessMode) -> void
    // Idempotent değildir — model/mod değişmedikçe tekrar çağrılmamalı.
    // RGA context/handle'ları burada bir kez kurulur.

process(source: DmaBuffer, mode_context: CropContext?) -> (DmaBuffer, CropTransform?)
    // mode_context: TargetCentricCrop modunda son bilinen bbox merkezi.
    //               FullFrameLetterbox modunda kullanılmaz (null).
    // Dönüş: yeni DmaBuffer (havuzdan), ve TargetCentricCrop modundaysa
    //        CropTransform (FullFrameLetterbox modunda null).

shutdown() -> void
    // RGA context'lerini serbest bırakır. Havuza dokunmaz (IBufferPool'un işi).
```

---

## 9. Açıkça Yapmadıkları

- Inference çağırmaz, `IInferenceEngine`'e referans tutmaz (sadece `ModelProfile`'ı bir kere alır, sonra bağımsızdır).
- NMS, bbox decode, peak-finding yapmaz (bunlar `IPostprocessor`'ın işi).
- Thread yönetmez, kendi kendine çağrılmaz — orkestratör tarafından tetiklenir.
- `pass_through` kararını NPU tarafında vermez, sadece kendi çıktısının dtype'ını doğru üretir.
