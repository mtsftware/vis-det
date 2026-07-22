# IPostprocessor — Yazılı Arayüz Sözleşmesi

> Kod değildir. `IInferenceEngine`'in ürettiği ham tensör çıktısını anlamlı sonuca (bbox, güven skoru) çeviren katmanın sözleşmesidir.

**Kaynaklar:** Rapor 2, Rapor 4, Rapor 6 + `01-IPreprocessor.md` (CropTransform bağımlılığı) + `02-IInferenceEngine.md` (rknn_tensor_attr alanları).

---

## 1. Sorumluluk Sınırı

`IPostprocessor`, `IInferenceEngine::getOutputs()`'un döndürdüğü ham tensör(ler)i alır, CPU üzerinde hafif matematiksel işlemlerle (dequant, decode, NMS veya peak-finding) yorumlar, `IPreprocessor`'ın ürettiği `CropTransform`'u kullanarak koordinatları **tam kare uzayına** geri çevirir.

**Bilir:** Çıktı tensörünün `rknn_tensor_attr` geometrisini (zaten `TensorResult` içinde taşınır), hangi model tipi için çalıştığını (Detector mi, TrackerHead mi — iki ayrı implementasyon).

**Bilmez:** RGA'nın nasıl çalıştığını, NPU çekirdek atamasını, buffer havuzunu.

---

## 2. İki Ayrı İmplementasyon — Ortak Arayüz, Farklı Mantık

### 2.1 `DetectorPostprocessor` (YOLO-tarzı)

Girdi: sınıf skorları + bbox regresyon tensörleri (letterbox uzayında, örn. 640×640).

Adımlar:
1. **Dequant (gerekirse):** `qnt_type == AFFINE_ASYMMETRIC` ve `want_float=0` alınmışsa: `x_float = (x_int8 - attr.zp) * attr.scale`. `want_float=1` ile alınmışsa bu adım `IInferenceEngine` tarafında zaten yapılmıştır, atlanır.
2. Kutu adaylarını çöz (grid → xywh → xyxy).
3. Güven eşiği filtresi.
4. NMS (Non-Maximum Suppression).
5. **Letterbox ters dönüşümü:** `IPreprocessor::process()`'ten gelen ölçek oranı (`ratio`) ve padding ofseti (`dw`, `dh`) kullanılarak kutular orijinal kare çözünürlüğüne geri ölçeklenir: `x_orig = (x_letterbox - dw) / ratio`.

Çıktı: `vector<Detection>{bbox, class_id, confidence}`.

### 2.2 `TrackerHeadPostprocessor` (Siamese-tarzı)

> **Not:** Bu implementasyonun mantığı 6 araştırma raporunda doğrudan işlenmedi; Siamese takip modellerinin (SiamRPN ailesi, NanoTrack, LightTrack) standart çıktı yorumlama pratiğinden sentezlenmiştir, RKNN'in dequant/tensör-geometri kurallarıyla birleştirilerek.

Girdi: `head` modelinin ürettiği bir **güven haritası** (confidence/classification map, küçük bir 2D grid) ve bir **bbox regresyon haritası**.

Adımlar:
1. Dequant gerekiyorsa aynı formül (§2.1 madde 1).
2. **Hanning penceresi ile cezalandırma:** Güven haritasına, merkeze yakın konumları ödüllendiren bir Hanning penceresi elementwise çarpılır — bu, hedefin bir önceki kareye göre çok uzağa "sıçramasını" cezalandırarak kararlılık sağlar (jitter azaltma).
3. **Peak-finding:** Cezalandırılmış haritada en yüksek skorlu hücre bulunur; bu hücrenin güven skoru `TrackResult.confidence` olur.
4. **Sub-pixel iyileştirme (opsiyonel, ileri seviye):** Peak hücresi etrafındaki komşu skorlardan ağırlıklı ortalama ile alt-piksel hassasiyetinde konum düzeltmesi.
5. Peak konumundaki bbox regresyon değerleri okunur (arama-bölgesi-lokal koordinatlarda).
6. **Kırpma ters dönüşümü (kesinlikle zorunlu):** `IPreprocessor`'dan gelen `CropTransform{origin_x, origin_y, scale}` ile: `x_full_frame = origin_x + (x_search_local × scale)`. Bu adım atlanırsa tracker'ın ürettiği koordinatlar anlamsızdır — arama bölgesi her karede farklı bir merkez etrafında kırpıldığı için (detector'ın sabit letterbox'ının aksine), bu dönüşüm **her çağrıda yeniden** hesaplanmalıdır.

Çıktı: `TrackResult{bbox, confidence}`.

---

## 3. Güven Eşiği ve "Hedef Kayıp" Sinyali

`TrackerHeadPostprocessor`, `confidence < threshold` durumunu **kendi içinde karar vermez** — sadece ham `confidence` değerini döndürür. Eşik karşılaştırması ve "hedefi kaybettik, detector'ı tetikle" kararı `ITracker`'ın sorumluluğundadır (bkz. `05-ITracker.md` §State Machine). Bu ayrım bilinçlidir: eşik değeri modelin kendisine değil, sistemin genel davranış politikasına ait bir parametredir.

---

## 4. Metod Sözleşmesi (pseudo-imza)

```
// DetectorPostprocessor
process(outputs: vector<TensorResult>, letterbox_info: LetterboxTransform)
    -> vector<Detection>

// TrackerHeadPostprocessor
process(outputs: vector<TensorResult>, crop_transform: CropTransform)
    -> TrackResult{bbox, confidence}
```

Her iki implementasyon da `IPostprocessor` arayüzünü paylaşır ama farklı ikinci parametre tipi alır (`LetterboxTransform` vs `CropTransform` — ikisi de `01-IPreprocessor.md`'den gelir, geometrik olarak benzer ama farklı anlam taşır: biri sabit letterbox oranı, diğeri her frame değişen dinamik kırpma).

---

## 5. Açıkça Yapmadıkları

- Inference çağırmaz.
- Buffer havuzu yönetmez.
- Eşik/state-machine kararı vermez (sadece ham sayısal sonucu üretir).
- RGA/NPU donanımına dokunmaz — tamamen CPU üzerinde, hafif matematiksel işlemler.
