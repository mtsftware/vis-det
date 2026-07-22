# ITracker — Yazılı Arayüz Sözleşmesi

> Kod değildir. Bu, sistemin en karmaşık arayüzüdür çünkü kendi başına bir "orkestratör" gibi davranır — birden fazla `IInferenceEngine`, `IPreprocessor`, `IPostprocessor` instance'ını kompoze eder. `NanoTrack`/`LightTrack` bu sözleşmeye uyan somut implementasyonlardır.

**Kaynaklar:** Rapor 6 (Asimetrik Yük Dağılımı — ana kaynak), Rapor 4 (precision), gerçek `rknn_api.h`, ve kullanıcı kararları: **ayrı .rknn dosyaları (backbone+head, template/search ayrı)**, **tek hedef**, **sabit şablon**.

---

## 1. Sorumluluk Sınırı

`ITracker`, "bir görüntüde bir hedefi başlat ve her karede takip et" sözleşmesini dışarıya tek bir basit arayüz olarak sunar, ama içeride **üç ayrı NPU modelini** yönetir. Diğer katmanlardan farklı olarak, `ITracker` kendi `IPreprocessor`/`IPostprocessor`/`IInferenceEngine` instance'larının **sahibidir**, onları dışarıdan almaz.

**Bilir:** Hedefin son bilinen konumunu, şablon özellik tensörünü (cache'lenmiş), kendi state machine'ini.

**Bilmez:** RTSP/decode katmanını, diğer tracker'ların var olup olmadığını (yok zaten — tek hedef), detector'ın iç mantığını (sadece onun `Detection` çıktısını tüketir).

---

## 2. Model Kompozisyonu — `TrackerModelBundle`

Kullanıcı kararı gereği (ayrı `.rknn` dosyaları), `ITracker` üç ayrı `IInferenceEngine` instance'ı besler:

| Rol | Ne zaman çalışır | Girdi | Çıktı |
|---|---|---|---|
| `template_backbone` | Sadece `initialize()` ve re-detect'te | Şablon crop (örn. 127×127) | Şablon özellik tensörü — **cache'lenir** |
| `search_backbone` | Her karede | Arama bölgesi crop (örn. 255×255) | Arama özellik tensörü |
| `head` | Her karede | (cache'lenmiş şablon özellik) + (bu karenin arama özelliği) | Güven haritası + bbox regresyonu |

> **Tasarım notu:** `TrackerModelBundle`, sabit "3 model" varsayımına kilitlenmez — bir `role → model_path` eşlemesi olarak tanımlanır. Eğer gerçek dönüştürme sürecinde `template_backbone` ve `search_backbone` tek bir dosyada birleşirse (aynı sabit girdi boyutunu paylaşıyorlarsa), bundle 2 role ile de çalışır. Bu esneklik bilinçlidir: kesin dosya sayısı henüz doğrulanmadı, arayüz buna göre kilitlenmedi.

`template_backbone` ile `search_backbone` **ağırlık paylaşımı yapmaz** (`RKNN_FLAG_SHARE_WEIGHT_MEM`/`rknn_dup_context` kullanılmaz) — her ikisi de ayrı `.rknn` dosyaları olarak derlendiği için (farklı sabit girdi boyutları), RKNN açısından tamamen bağımsız modellerdir, kaynak ağırlıklar aynı olsa bile (bkz. `02-IInferenceEngine.md` §8).

---

## 3. NPU Çekirdek Ataması (Bu Turda Netleşen Üçlü Model)

Önceki (Rapor 6) ikili model — "Tracker→izole çekirdek, Detector→kombine çekirdek" — artık üçe genişliyor, çünkü **sabit şablon** kararı `template_backbone`'u "her frame kritik" değil, "**olay-tetiklemeli, seyrek**" kategorisine sokuyor — tıpkı detector gibi:

| Rol | Core mask | Gerekçe |
|---|---|---|
| `search_backbone` + `head` | `RKNN_NPU_CORE_2` (izole) | Her karede çalışır, gecikmesi kutsal, hiçbir zaman paylaşılmaz |
| `detector` + `template_backbone` | `RKNN_NPU_CORE_0_1` (kombine) | İkisi de seyrek çalışır **ve birbirini takip eder** (önce detector hedefi bulur, sonra template_backbone o crop'tan şablon çıkarır) — asla eşzamanlı değiller, bu yüzden aynı çekirdek çiftini güvenle paylaşabilirler |

Bu atama, `RKNN_NPU_CORE_AUTO` kullanmadan, yalnızca belgelenmiş `rknn_core_mask` değerleriyle kurulur (bkz. `02-IInferenceEngine.md` §6).

---

## 4. İki Farklı Preprocess Profili

`ITracker`, `IPreprocessor`'ı iki farklı `PreprocessMode::TargetCentricCrop` alt-profiliyle çağırır (bkz. `01-IPreprocessor.md` §3.2):

- `TemplateProfile` — sadece `initialize()` içinde, dar bağlam çarpanıyla.
- `SearchProfile` — her `track()` çağrısında, geniş bağlam çarpanıyla, merkez = **son bilinen bbox**.

---

## 5. State Machine

```
        initialize(frame, bbox)
              │
              ▼
        ┌──────────┐
        │ TRACKING │◄──────────────┐
        └────┬─────┘               │
             │ confidence < eşik   │ detector başarıyla
             ▼                     │ yeniden buldu →
        ┌──────────┐   detector    │ initialize() tekrar
        │   LOST   │──tetiklenir──►│ çağrılır
        └──────────┘               │
```

- **`TRACKING`:** `track()` her karede çağrılır, `search_backbone`+`head` çalışır. Sonuç güven skoru eşiğin üzerindeyse konum güncellenir.
- **Eşik altına düşünce:** `ITracker`, orkestratöre "hedef kayıp" sinyali verir (`is_target_lost` benzeri bir durum). **`template_backbone` bu noktada çalışmaz** — sadece durum değişir.
- **`LOST`:** `search_backbone`/`head` çağrılmaz (CPU/NPU boşa harcanmaz). Orkestratör, detector'ı tetikler.
- **Detector başarılı olursa:** `ITracker::initialize(yeni_frame, detector_bbox)` tekrar çağrılır → `template_backbone` bu crop üzerinde bir kez çalışır, yeni şablon cache'lenir, durum `TRACKING`'e döner.

**Sabit şablon kuralı (bu turun kararı):** `template_backbone`, `TRACKING` durumu boyunca **asla** yeniden çalışmaz, ne periyodik ne adaptif. Yalnızca `initialize()` çağrısında. Bu, `ITracker` arayüzünde ayrı bir `refreshTemplate()` metoduna **şu an gerek olmadığı** anlamına gelir — ama arayüz, gelecekte eklenebilecek bir genişletme noktası olarak bunu tamamen kapatmaz (bkz. §8).

---

## 6. Detector → Tracker Devir Teslim Varsayımı (Açıkça İşaretlenmiş)

Tek hedef takibi kararı verildiği için, `ITracker`'ın `LOST` durumundan çıkışı, detector'ın ürettiği aday kutulardan **hangisinin** "kaybedilen hedef" olduğuna karar vermeyi gerektirir. Bu, hiçbir raporda ele alınmadı ve şu an **açık bir varsayımla** ilerliyoruz:

> **Varsayım:** Detector birden fazla aday döndürürse, son bilinen hedef konumuna en yakın (IoU veya merkez mesafesi) olan aday seçilir. Hiç önceki konum yoksa (sistem başlangıcı), en yüksek güven skorlu aday seçilir.

Bu, `ITracker`'ın değil, orkestratörün (`06-system-architecture.md`'deki Detector Thread) sorumluluğunda bir seçim fonksiyonu olarak tasarlanmalı — `ITracker::initialize()` zaten seçilmiş tek bir `bbox` alır, seçim mantığını kendisi yapmaz. **Bu varsayım yanlışsa (örn. gerçek bir re-ID/appearance-matching gerekiyorsa), sadece orkestratördeki bu seçim fonksiyonu değişir, `ITracker` sözleşmesi etkilenmez** — ayrıştırma bilinçli yapıldı.

---

## 7. Tek Hedef Tasarımının Basitleştirdikleri

- `ITracker` bir **singleton nesne**dir — pipeline başına bir tane, havuzlanmaz, çoğaltılmaz.
- `IBufferPool`'da hedef-başına ayrı profil gerekmez (tek `TrackerTemplateInputPool`/`TrackerSearchInputPool` yeterli).
- Çoklu-nesne veri ilişkilendirme (Hungarian algorithm, ID yönetimi) **gerekmiyor** — bu önemli bir karmaşıklık tasarrufu.

---

## 8. Metod Sözleşmesi (pseudo-imza)

```
initialize(frame: DmaBuffer, bbox: BBox) -> void
    // TemplateProfile ile crop → template_backbone.run() → şablon cache'le.
    // Durumu TRACKING yap. template_backbone/detector'ın CORE_0_1'i burada
    // kısa süreliğine meşgul eder (sıralı, search_backbone'u etkilemez).

track(frame: DmaBuffer) -> TrackResult
    // SearchProfile ile crop (merkez=son bbox) → search_backbone.run()
    //   → head.run(cached_template, search_feat) → postprocess.
    // confidence < eşik ise durum LOST'a düşer, orkestratöre sinyal verilir.
    // CORE_2 üzerinde, tek thread'de, seri çalışır — asla paralelleştirilmez
    // (bkz. §9, Rapor 6'nın "durumlu modeller paralelleştirilemez" bulgusu).

getState() -> TrackerState { TRACKING, LOST }

shutdown() -> void
    // Üç IInferenceEngine instance'ını da kapatır.
```

---

## 9. Kritik Threading Kuralı (Rapor 6'dan, Değişmeden Geçerli)

`track()`'in çağrıldığı thread **tek ve serdir**. Ardışık karelerin round-robin şekilde farklı çekirdeklere/thread'lere dağıtılması (pipeline parallelism) tracker için **kavramsal olarak imkansızdır** — Frame N+1'in işlenmesi Frame N'in ürettiği "son bilinen konum"a bağımlıdır. Bu tekniği (Rapor 1'in genel önerisi, Rapor 6'nın YOLO gibi durumsuz modeller için doğruladığı) sadece **detector** için düşünebiliriz, tracker için asla.

---

## 10. Açıkça Yapmadıkları

- RTSP/decode/RGA donanım çağrılarını doğrudan yapmaz — sahip olduğu `IPreprocessor` instance'ları üzerinden delege eder.
- Detector'ın kendi mantığını (NMS, sınıf filtresi) bilmez — sadece `Detection` tipini tüketir.
- Çoklu hedef yönetmez (bu turun kararı gereği).
- Şablonu periyodik güncellemez (bu turun kararı gereği) — ama bu, mimari olarak kapatılmış bir kapı değil, şu anki kapsam dışı bırakılmış bir özellik.
