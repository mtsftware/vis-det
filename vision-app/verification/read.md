# best-rk3588.rknn — Model Bilgisi (cihazda dogrulandi)

Kaynak: `vision-app/models/best-rk3588.rknn`
Sorgu araci: `verification/model_info_tool.cpp` (firefly RK3588 kartinda derlenip
calistirildi, cikti asagida).

## SDK / Driver

- RKNN API version: `1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)`
- Driver version: `0.9.3`
- Export toolkit: RKNN-Toolkit2 `2.3.2`, hedef platform `rk3588`, kaynak framework `ONNX`

## Girdi tensor: `images`

| Alan | Deger |
|---|---|
| n_dims | 4 |
| dims | `(1, 640, 640, 3)` — **NHWC sirasi** |
| fmt | `NHWC` |
| type | `INT8` |
| qnt_type | `AFFINE` (asymmetric) |
| zero_point | `-128` |
| scale | `0.00392157` (=1/255) |
| size / size_with_stride | 1.228.800 bayt (stride yok, w_stride=640=width) |

**Onemli:** `rknn_query` `INPUT_ATTR` (mantiksal) ile `NATIVE_INPUT_ATTR` (NPU'nun
ic donanim formati) BIREBIR ayni cikti — yani bu model icin ekstra bir
donusum katmani yok, native format da NHWC/INT8.

**Preprocessing formulu:** `scale=1/255, zp=-128` -> quant formulu
`q = round(x/255 / scale) + zp = round(x) - 128` (x: 0-255 arasi ham piksel).
Yani pratikte: RGB uint8 goruntuyu al, **her byte'tan 128 cikar, int8'e cast
et** (mean=[0,0,0], std=[255,255,255] ile normalize edip int8 kuantize etmeye
denk). `rknn_inputs_set` ile UINT8/NHWC ham piksel verip `pass_through=0`
birakmak da (tracking-app'teki `RknnInferenceEngine::setInputCopy` deseni)
gecerli — runtime donusumu kendi icinde yapar.

## Cikti tensor: `output0`

| Alan | Deger |
|---|---|
| n_dims | 3 |
| dims | `(1, 5, 8400)` |
| fmt | `UNDEFINED` (3 boyutlu tensor icin NCHW/NHWC ayrimi anlamsiz) |
| type | `INT8` |
| qnt_type | `AFFINE` (asymmetric) |
| zero_point | `-125` |
| scale | `2.60278153` |
| size | 42.000 bayt |

**Dequant formulu:** `gercek_deger = (int8_deger - zero_point) * scale
= (int8_deger + 125) * 2.60278153`

**Yapi:** YOLOv8n, tek sinif (`nc=1`) fine-tune. `5 = 4 (box: cx,cy,w,h) + 1
(sinif skoru)`, `8400 = 80x80 + 40x40 + 20x20` (640x640 girdi icin uc olcek
toplami). DFL (Distribution Focal Loss) box decode ve concat/transpose islemi
ONNX->RKNN donusumu sirasinda grafige gomulmus (model iceresinde
`model.22.dfl.*`, `Softmax`, `Sigmoid` düğümleri var — bkz. `strings
models/best-rk3588.rknn`), yani runtime çıktısı ham DFL dagilimlari degil,
zaten **kutu + skor** olarak gelen tek bir tensor. Skor kanalinin sigmoid
uygulanmis mi geldigi (grafikte `Sigmoid` node'u var) fiili bir inference
calistirilip (bilinen bir girdiyle) cikan skor degerinin [0,1] araliginda
olup olmadigina bakilarak dogrulanmali — bu tool sadece statik tensor
attr'lerini okur, gercek bir `rknn_run` yapmaz.

## Bellek

- `total_weight_size`: 3.452.992 bayt (~3.3 MB, agirliklar)
- `total_internal_size`: 6.560.000 bayt (~6.3 MB, ara aktivasyonlar)
- `total_dma_allocated_size`: 11.243.520 bayt (~10.7 MB)

## Inference icin ozet checklist

1. Girdi: `640x640x3`, **NHWC**, RGB (rgb2bgr=False export config'inde), UINT8
   piksel verilebilir (`pass_through=0`, runtime int8'e kendi ceviriyor) ya da
   elle `pixel-128` yapip INT8 verilebilir.
2. `rknn_inputs_set` -> `rknn_run` -> `rknn_outputs_get` (`want_float=1`
   verirsen runtime dequant'i senin icin yapar, `float32 (1,5,8400)` alirsin —
   tracking-app'teki `RknnInferenceEngine::getOutput` deseniyle ayni).
3. `want_float=0` ile ham INT8 alirsan yukaridaki `(v+125)*2.60278153`
   formuluyle kendin dequant etmen lazim.
4. Cikti `(5, 8400)`: satir 0-3 = box (cx,cy,w,h — 640x640 piksel uzayinda,
   NMS/olcekleme sonrasi orijinal frame'e geri map edilmeli), satir 4 = tek
   sinif skoru.
5. Post-process: skor esiklemesi + NMS (coco/anchor-free YOLOv8 standardi) —
   tracking-app'te hazir bir postprocess yok, bu adim vision-app'e ozel
   yazilacak.
