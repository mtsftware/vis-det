# IInferenceEngine — Yazılı Arayüz Sözleşmesi

> Kod değildir. `librknn_api/include/rknn_api.h` (gerçek header, 721 satır) doğrudan incelenerek yazılmıştır. Araştırma raporlarının iddia ettiği ama bu header'da bulunmayan semboller (`rknn_inputs_map`, `RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE` vb.) bu sözleşmede **kullanılmamıştır**.

**Kaynaklar:** Rapor 2, Rapor 4, Rapor 5, Rapor 6 + gerçek `rknn_api.h` / `rknn_matmul_api.h`.

---

## 1. Sorumluluk Sınırı

`IInferenceEngine`, **tek bir yüklü `.rknn` modelini** (`tek rknn_context`) saran, precision-farkında, zero-copy'ye zorunlu bir wrapper'dır. Sistemde birden fazla `IInferenceEngine` instance'ı olacaktır (detector, tracker'ın search_backbone'u, template_backbone'u, head'i — bkz. `05-ITracker.md`).

**Bilir:** Kendi modelinin giriş/çıkış tensör geometrisini (`rknn_query` ile), kendi precision profilini, kendisine atanmış NPU çekirdek maskesini.

**Bilmez:** RGA'nın nasıl çalıştığını, hangi buffer havuzundan geldiğini, çıktının nasıl yorumlanacağını (postprocess mantığı), diğer `IInferenceEngine` instance'larının varlığını.

---

## 2. Gerçek RKNN API Akışı (Header'dan Doğrulanmış)

Araştırma raporlarının bahsettiği `rknn_inputs_map`/`rknn_outputs_map`/`rknn_inputs_sync`/`rknn_outputs_sync` fonksiyonları **bu SDK'da yok**. Gerçek zero-copy akışı:

```
1. rknn_init(&ctx, model_data, size, RKNN_FLAG_MEM_ALLOC_OUTSIDE, NULL)
2. rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num))
3. her giriş/çıkış için:
     rknn_query(ctx, RKNN_QUERY_INPUT_ATTR / RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr))
   → ModelProfile bu sorgulardan inşa edilir (§4)
4. rknn_set_core_mask(ctx, core_mask)   // ASLA RKNN_NPU_CORE_AUTO (§5)
5. Her frame:
   a. rknn_create_mem_from_fd(ctx, fd, virt_addr, size, offset)
      // alternatif: rknn_create_mem_from_phys() ya da MPP buffer'ları için
      //             rknn_create_mem_from_mb_blk() — RGA çıktısı söz konusu
      //             olduğunda fd yolu esastır.
   b. rknn_set_io_mem(ctx, mem, &attr)
   c. rknn_run(ctx, &run_extend)   // run_extend.fence_fd / non_block kullanılabilir (§6)
   d. (non_block ise) rknn_wait(ctx, &run_extend)
   e. rknn_outputs_get(ctx, n_outputs, outputs, &output_extend)
      // outputs[i].want_float precision'a göre ayarlanır (§7)
   f. rknn_outputs_release(ctx, n_outputs, outputs)
6. rknn_destroy(ctx)
```

`rknn_set_io_mem` **tek bağlama çağrısıdır** — ayrı bir "map" adımı yoktur. Bu, önceki tur analizlerimizde varsayılan "map bir kere, sync her frame" modelinden **farklıdır**; o model daha yeni bir SDK sürümünü tarif ediyordu.

---

## 3. `rknn_tensor_attr` — Gerçek Alanlar (Header'dan Birebir)

| Alan | Tip | Erişim | Anlamı |
|---|---|---|---|
| `w_stride` | `uint32_t` | Salt okunur | Genişlik adımı. **0 ise gerçek genişliğe eşittir.** |
| `h_stride` | `uint32_t` | Salt yazılır | Yükseklik adımı. 0 set edilirse gerçek yüksekliğe eşit sayılır. |
| `size_with_stride` | `uint32_t` | Salt okunur | Stride dahil brüt bellek boyutu — **buffer tahsisi bu değere göre yapılır.** |
| `zp` | `int32_t` | — | Sıfır noktası (yalnızca `RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC`). |
| `scale` | `float` | — | Ölçek çarpanı. Dequant formülü: `x_float = (x_int8 - zp) * scale`. |
| `pass_through` | `uint8_t` | — | TRUE → veri dönüşümsüz doğrudan girer. FALSE → `type`/`fmt`'e göre dönüştürülür. |
| `type` | `rknn_tensor_type` | — | `RKNN_TENSOR_FLOAT16` / `INT8` / `UINT8` / ... |
| `qnt_type` | `rknn_tensor_qnt_type` | — | `RKNN_TENSOR_QNT_NONE` / `RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC` / `RKNN_TENSOR_QNT_DFP` |
| `fmt` | `rknn_tensor_format` | — | `NCHW` / `NHWC` / `NC1HWC2` |

---

## 4. Precision Profili — Runtime Otomatik Tespit

`initialize()` tamamlandıktan hemen sonra, **hiçbir harici config'e gerek kalmadan**, sadece `rknn_query` çıktısına bakarak:

```
IF attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
    precision = INT8   (attr.type genelde INT8 ya da UINT8; attr.zp/scale geçerli)
ELIF attr.type == RKNN_TENSOR_FLOAT16 AND attr.qnt_type == RKNN_TENSOR_QNT_NONE:
    precision = FP16   (attr.zp/scale geçersiz/kullanılmaz)
```

Bu tespit `ModelProfile.precision` alanını doldurur ve aşağıdaki üç kararı **otomatik olarak** kilitler — hiçbiri dışarıdan set edilemez:

1. `pass_through` değeri (§5 — asla elle set edilmez)
2. `want_float` stratejisi (§7)
3. `IPreprocessor`'a iletilecek hedef dtype (bkz. `01-IPreprocessor.md` §5)

---

## 5. `pass_through` — Sabit, Otomatik, Asla Elle Değiştirilemez

| Precision | `pass_through` |
|---|---|
| INT8/UINT8 | `1` (TRUE) güvenli — giriş zaten 8-bit alanda |
| FP16 | `0` (FALSE) **zorunlu**, `IPreprocessor` FP16 dönüşümünü önceden yapmışsa |

**Kesin yasak:** FP16 modelde `pass_through=1` ile ham UINT8 RGB verisi vermek. NPU, iki bitişik UINT8 baytını IEEE754 FP16 bit deseni olarak yorumlar → segfault YOK, `RKNN_SUCC` döner, çıktı tamamen çöp (Rapor 4, "bayt yozlaşması"). Bu alan `ModelProfile`'ın bir parçası olarak **derive edilir**, `IInferenceEngine` arayüzünde public bir setter'ı yoktur.

---

## 6. NPU Çekirdek Ataması — Sadece Belgelenmiş Değerler

Gerçek `rknn_core_mask` enum'u:

```
RKNN_NPU_CORE_AUTO = 0    (varsayılan — KULLANILMAYACAK)
RKNN_NPU_CORE_0    = 1
RKNN_NPU_CORE_1    = 2
RKNN_NPU_CORE_2    = 4
RKNN_NPU_CORE_0_1     = 0|1  = 3   (belgelenmiş kombine)
RKNN_NPU_CORE_0_1_2   = 0_1|2 = 7  (belgelenmiş kombine)
```

**`RKNN_NPU_CORE_1_2` diye bir değer YOKTUR.** Bitwise olarak elle üretilebilir (2|4=6) ama Rockchip bunu resmi olarak tanımlamamış/test etmemiştir — kullanılmaz.

**Kesin kural:** `RKNN_NPU_CORE_AUTO` hiçbir `IInferenceEngine` instance'ında kullanılmaz. Sebep: sürücü, AUTO modda aynı fiziksel çekirdeğe düşen iki context'i spinlock ile serileştirir; bu, RKNPU 0.9.2 gibi sürücülerde spinlock recursion → kernel panic'e yol açtığı resmi olarak raporlanmıştır (Rapor 6, Issue #329).

**Sistemdeki nihai atama** (bkz. `06-system-architecture.md` §NPU Çekirdek Tablosu için tam gerekçe):

| Rol | Core mask | Neden |
|---|---|---|
| `search_backbone` + `head` (tracker, her frame) | `RKNN_NPU_CORE_2` | İzole, hiçbir zaman dokunulmaz — gecikme deterministik kalır |
| `detector` + `template_backbone` (olay-tetiklemeli, ardışık) | `RKNN_NPU_CORE_0_1` | İkisi de seyrek ve **birbirini takiben** çalışır (asla eşzamanlı değil), kombine modun DMA senkronizasyon overhead'i burada tolere edilebilir |

---

## 7. `want_float` Stratejisi (`rknn_output.want_float`, `uint8_t`)

| Precision | `want_float` | Maliyet | Gerekçe |
|---|---|---|---|
| FP16 | `1` (her zaman) | Neredeyse bedava (bit-genişletme, ARM NEON) | Zaten yarı-hassasiyetli float, dequant zinciri yok |
| INT8, dikkat/softmax içeren model (ör. tracker head) | `1` | Pahalı (CPU'da `(x-zp)*scale` her eleman için) ama zorunlu | `want_float=0` → 8-bit çözünürlük softmax'ı çökertir, "Nicemleme Çöküşü" (Rapor 4) |
| INT8, basit konvolüsyon-ağırlıklı model | `0` mümkün (profiling sonrası) | Düşük | Risk daha az, ama varsayılan yine `1` başlanmalı |

**Varsayılan:** `want_float=1`, her `IInferenceEngine` instance'ı için. `0`'a düşürmek yalnızca ölçülmüş bir darboğaz sonrası, model bazında bilinçli bir optimizasyon kararıdır.

---

## 8. Kullanılabilir Flag'ler (Gerçek Header — Tam Liste)

| Flag | Değer | Kullanım kararı |
|---|---|---|
| `RKNN_FLAG_MEM_ALLOC_OUTSIDE` | `0x10` | **Her zaman** — zero-copy'nin ön koşulu |
| `RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE` | `0x200` | Kullanılmaz (varsayılan iç yönetime bırakılır — SRAM taşma riski, §9) |
| `RKNN_FLAG_SHARE_WEIGHT_MEM` | `0x20` | Kullanılmaz — bizim modellerimizin hiçbiri aynı `.rknn` dosyasının kopyası değil (template_backbone ve search_backbone bile farklı sabit girdi boyutuyla **ayrı dosyalar** olarak derlenmiş, bkz. `05-ITracker.md`) |
| `RKNN_FLAG_ASYNC_MASK` | `0x04` | Kullanılmaz — çok-thread'li mimarimizde (her context kendi thread'inde bloklanarak bekler) gerek yok, ek karmaşıklık getirir |
| `RKNN_FLAG_COLLECT_PERF_MASK` | `0x08` | Geliştirme/profiling build'lerinde açık, üretimde kapalı |
| `RKNN_FLAG_FENCE_IN_OUTSIDE` / `_OUT_OUTSIDE` | `0x40` / `0x80` | **İleri seviye, opsiyonel** — RGA'nın "yazma bitti" sinyalini `rknn_run_extend.fence_fd`'ye bağlamak için değerlendirilebilir (§10) |
| `RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU` | `0x400` | **Tracker'ın `head` modeli için önerilir** — korelasyon operasyonu NPU'da tam desteklenmezse CPU yerine GPU'ya düşsün |
| `RKNN_FLAG_COLLECT_MODEL_INFO_ONLY` | `0x100` | Kullanılmaz (runtime davranışı değil, tanılama amaçlı) |

**Kesin yasak:** `RKNN_INTERNAL_MEM_TYPE=sram` / `RKNN_WEIGHT_MEM_TYPE=sram` ortam değişkenleri. Büyük modelin (detector) aktivasyonları 32KB'lık L1 SRAM scratchpad'e sığmadığında `REGTASK Overflow (0xe010)` ile sistem kilitlenir (Rapor 6).

---

## 9. Yeni Bulgu: Native Fence Desteği (Hiçbir Raporda Yok)

`rknn_run_extend` struct'ı:
```
frame_id     : uint64_t   (çıktı — hangi kareye ait sonuç)
non_block    : int32_t    (0=blocking, 1=non-blocking)
timeout_ms   : int32_t
fence_fd     : int32_t    (dışarıdan gelen dma-fence)
```

Bu, Rapor 5'in "Android'den ilham alıp explicit fence taklit et" önerisinin **native karşılığıdır**. `IPreprocessor`'ın RGA çağrısı bir fence fd üretebiliyorsa (RGA API'sinin senkron/asenkron moduna bağlı), bu doğrudan `rknn_run`'a geçirilip NPU'nun RGA'nın bitişini CPU'da beklemeden algılaması sağlanabilir. **Durum: değerlendirilecek, ilk sürümde zorunlu değil** — RGA çağrıları zaten senkron (blocking) kullanılıyorsa bu alan `fence_fd=-1` ile boş bırakılabilir.

---

## 10. Metod Sözleşmesi (pseudo-imza)

```
initialize(model_path: string, core_mask: CoreMask) -> void
    // rknn_init + RKNN_FLAG_MEM_ALLOC_OUTSIDE, ardından rknn_set_core_mask.
    // AUTO asla geçirilmez — CoreMask enum'unda AUTO seçeneği yoktur.

queryProfile() -> ModelProfile
    // rknn_query(IN_OUT_NUM) + her tensör için INPUT_ATTR/OUTPUT_ATTR.
    // §4'teki tespiti yapar, pass_through ve want_float kararlarını kilitler.
    // initialize()'dan hemen sonra, process() çağrılmadan önce zorunlu.

bindInput(mem: DmaBuffer, tensor_index: uint32) -> void
    // rknn_create_mem_from_fd + rknn_set_io_mem.

run(fence_fd: int32 = -1) -> void
    // rknn_run. fence_fd verilirse RKNN_FLAG_FENCE_IN_OUTSIDE aktifse kullanılır.

getOutputs() -> vector<TensorResult>
    // rknn_outputs_get (want_float profile'a göre) + rknn_outputs_release.

shutdown() -> void
    // rknn_destroy.
```

---

## 11. Açıkça Yapmadıkları

- RGA/letterbox/crop mantığı bilmez (girdi olarak hazır `DmaBuffer` bekler).
- NMS, peak-finding, bbox decode yapmaz.
- Diğer `IInferenceEngine` instance'larının farkında değildir — `ITracker`/orkestratör bu koordinasyonu yapar.
- Kendi thread'ini açmaz; hangi thread'den çağrıldıysa o thread'de bloklar.
