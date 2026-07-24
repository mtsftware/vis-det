"""
inference.py — RKNN NPU model runner for RK3588 / RK3576 boards.

Drop-in replacement for the original TFLite version.
Same public API → main.py needs only ONE change (dequantization guard).

Expected model format  (airockchip RKNN-optimised YOLOv8 export)
─────────────────────────────────────────────────────────────────
9 output tensors:  box_s0, score_s0, sum_s0,
                   box_s1, score_s1, sum_s1,
                   box_s2, score_s2, sum_s2

  box_s   : [1, 64, H, W]   4 sides × REG_MAX=16 DFL bins  NCHW  float32
  score_s : [1, nc, H, W]   class probabilities, sigmoid applied  NCHW  float32
  sum_s   : ignored

3 scales for a 640×640 input:
  s=0 stride 8  → H=W=80
  s=1 stride 16 → H=W=40
  s=2 stride 32 → H=W=20

Preprocessing: letterbox with grey (114,114,114) padding — must match the
letterbox used during RKNN model export (same as RKNNDetector.cpp).

References
──────────
  RKNNDetector.cpp  — C++ ground-truth implementation shipped with this repo
  rknn-toolkit2 / rknnlite Python API documentation
"""


from __future__ import annotations

import atexit
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Model constants ────────────────────────────────────────────────────────────
RKNN_INPUT_SIZE = 640     # YOLOv8 standard square input
N_SCALES        = 3
STRIDES         = (8, 16, 32)
REG_MAX         = 16      # DFL distribution bins

# Shape debug — printed once on the first inference call
_shape_logged: bool = False


# =============================================================================
# RKNN runtime import
# =============================================================================

def _import_rknn() -> tuple:
    """
    Import the correct RKNN Python binding.

    RKNNLite (rknnlite)        -> on-device runtime  (pre-installed on board)
    RKNN SDK (rknn-toolkit2)   -> dev-host simulator / remote-target

    Returns (RKNNClass, is_lite: bool).

    Hata ayırt etme:
      ImportError("No module named 'rknnlite'") → paket kurulmamış → rknn.api'ye geç
      ImportError("librknnrt.so: cannot open…")  → paket kurulu ama .so eksik → açık hata ver
    """
    _rknnlite_err = None

    try:
        from rknnlite.api import RKNNLite
        logger.debug("RKNN: using rknnlite (on-device NPU runtime)")
        return RKNNLite, True
    except ImportError as e:
        _rknnlite_err = e

    # rknnlite kurulu ama shared library (librknnrt.so) bulunamıyor — .so hatasını
    # "No module named" hatasından ayırt et. .so eksikse rknn.api'ye geçme, hata ver.
    if "No module named" not in str(_rknnlite_err):
        raise ImportError(
            "rknnlite kurulu fakat RKNN runtime shared library bulunamadı.\n"
            f"  Hata : {_rknnlite_err}\n"
            "\n"
            "  Çözüm 1 — library path ekle (aktif oturumda):\n"
            "    export LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH\n"
            "\n"
            "  Çözüm 2 — kalıcı ldconfig:\n"
            "    sudo ldconfig /usr/lib\n"
            "\n"
            "  Çözüm 3 — library var mı kontrol et:\n"
            "    find /usr -name 'librknnrt.so*' 2>/dev/null\n"
            "    find /opt -name 'librknnrt.so*' 2>/dev/null"
        )

    # rknnlite gerçekten kurulmamış — dev host için rknn-toolkit2'yi dene
    try:
        from rknn.api import RKNN
        logger.debug("RKNN: using rknn-toolkit2 (dev-host simulator)")
        return RKNN, False
    except ImportError:
        raise ImportError(
            "RKNN Python runtime bulunamadı.\n"
            "  Board üzerinde:\n"
            "    pip install rknn_toolkit_lite2-2.3.2-cp38-cp38-"
            "manylinux_2_17_aarch64.manylinux2014_aarch64.whl\n"
            "  Dev host:\n"
            "    pip install rknn-toolkit2"
        )


# =============================================================================
# Label loading  (unchanged from TFLite version)
# =============================================================================

def load_labels(labels_path: str) -> list[str]:
    """Load class labels from a text file (one label per line)."""
    path = Path(labels_path)
    if not path.exists():
        # Also try relative to this file's directory
        alt = Path(__file__).parent / labels_path
        if alt.exists():
            path = alt
        else:
            logger.warning(f"Labels file not found: {labels_path}  ->  using numeric IDs.")
            return []
    with path.open() as f:
        return [ln.strip() for ln in f if ln.strip()]


# =============================================================================
# Pipeline loading
# =============================================================================

def load_pipeline(config: dict) -> list[dict]:
    """
    Load the RKNN model and return a single-element pipeline list.

    The list is compatible with the TFLite pipeline interface used by
    main.py (same keys: label, input_details, output_details, is_npu, avg_ms).

    input_details[0]["shape"]  ->  [1, H, W, C]  NHWC  so main.py's shape
    queries (input_h / input_w) work without any modification.
    """
    raw_path = config["model"]["path"]

    # Resolve relative paths -> script directory first, then cwd
    if not Path(raw_path).is_absolute():
        candidate = Path(__file__).parent / raw_path
        model_path = str(candidate) if candidate.exists() else raw_path
    else:
        model_path = raw_path

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"RKNN model not found: {model_path}\n"
            f"  -> Place '{Path(raw_path).name}' in the same directory as inference.py\n"
            f"     or set an absolute path in config.json  (model.path)."
        )

    RKNNClass, is_lite = _import_rknn()
    rknn = RKNNClass()

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"rknn.load_rknn failed (ret={ret}): {model_path}")

    if is_lite:
        # On-device: let the runtime pick the optimal NPU core
        ret = rknn.init_runtime(core_mask=RKNNClass.NPU_CORE_AUTO)
    else:
        # Dev-host simulator; add rknn_target to config["model"] to target a
        # real board over USB/network, e.g. "rknn_target": "rk3588"
        target = config.get("model", {}).get("rknn_target", None)
        ret = rknn.init_runtime(target=target)

    if ret != 0:
        raise RuntimeError(f"rknn.init_runtime failed (ret={ret})")

    # Ensure release on interpreter exit (avoids NPU resource leak)
    atexit.register(rknn.release)

    logger.info(
        "RKNN model loaded  : %s\n"
        "  Input shape      : [1, %d, %d, 3]  NHWC  uint8\n"
        "  Runtime          : %s",
        Path(model_path).name,
        RKNN_INPUT_SIZE, RKNN_INPUT_SIZE,
        "RKNNLite (NPU)" if is_lite else "RKNN SDK (dev-host)"
    )

    # Fake TFLite-compatible input_details so main.py shape queries work unchanged.
    # Shape is NHWC [batch, H, W, channels] — main.py reads [1] for H, [2] for W.
    input_details = [{
        "index":        0,
        "shape":        np.array([1, RKNN_INPUT_SIZE, RKNN_INPUT_SIZE, 3], dtype=np.int32),
        "dtype":        np.uint8,
        "quantization": (1.0, 0),   # not used for RKNN
        "name":         "images",
    }]

    stage = {
        "label":          "rknn",
        "rknn":           rknn,
        "input_details":  input_details,
        "output_details": [],     # RKNN returns outputs as a plain list
        "is_npu":         True,
        "avg_ms":         0.0,
    }
    return [stage]


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_frame(frame, input_details: list) -> np.ndarray:
    """
    Letterbox-resize the frame to the RKNN model's input size.

    Letterbox (aspect-ratio-preserving resize with grey padding) matches
    RKNNDetector::letterbox() in C++ exactly, so coordinate un-projection
    in postprocess_detections() produces correct pixel coordinates.

    Padding colour: (114, 114, 114) — matches YOLOv8 RKNN export convention.

    Args:
        frame         : BGR uint8 numpy array  (H x W x 3)
        input_details : pipeline input_details list (shape -> target H/W)

    Returns:
        tensor : uint8 ndarray  [1, target_h, target_w, 3]  RGB
    """
    import cv2

    shape    = input_details[0]["shape"]
    target_h = int(shape[1])   # NHWC -> H
    target_w = int(shape[2])   # NHWC -> W

    fh, fw = frame.shape[:2]
    r  = min(target_h / fh, target_w / fw)
    nw = round(fw * r)
    nh = round(fh * r)
    dw = (target_w - nw) // 2
    dh = (target_h - nh) // 2

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[dh : dh + nh, dw : dw + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0)   # [1, H, W, 3]  uint8


# =============================================================================
# Stage invoke  (called by main.py's _stage_loop thread)
# =============================================================================

def invoke_stage(stage: dict,
                 input_tensors: list) -> tuple:
    """
    Run one RKNN inference pass.

    main.py calls this as:
        out_data, elapsed_ms = inference.invoke_stage(stage, data)
    and later reads:
        raw = raw_list[0]

    We return  [rknn_outputs_list]  so that  raw_list[0]  yields the list
    of 9 tensors, which postprocess_detections() detects and routes to the
    RKNN decoder.

    Args:
        stage         : pipeline stage dict (contains 'rknn' handle)
        input_tensors : [uint8_nhwc_tensor]  shape  [1, 640, 640, 3]

    Returns:
        outputs_wrapped : [[tensor0, ..., tensor8]]
        elapsed_ms      : wall-clock invoke time  (ms)
    """
    rknn = stage["rknn"]

    t_start      = time.monotonic()
    rknn_outputs = rknn.inference(inputs=[input_tensors[0]])
    elapsed_ms   = (time.monotonic() - t_start) * 1000

    if rknn_outputs is None:
        logger.error("rknn.inference() returned None — check model / runtime init")
        rknn_outputs = []

    return [rknn_outputs], elapsed_ms   # raw_list[0] = list of 9 float32 tensors


# =============================================================================
# Internal helpers
# =============================================================================

def _lb_params(frame_h: int, frame_w: int,
               target_h: int = RKNN_INPUT_SIZE,
               target_w: int = RKNN_INPUT_SIZE) -> tuple:
    """
    Reproduce preprocess_frame()'s letterbox arithmetic.

    Returns (ratio, dw, dh) — same variables used in C++ letterbox():
        ratio : resize scale applied to the original frame
        dw, dh: horizontal/vertical padding added to each side (pixels)
    """
    r  = min(target_h / frame_h, target_w / frame_w)
    nw = round(frame_w * r)
    nh = round(frame_h * r)
    dw = (target_w - nw) // 2
    dh = (target_h - nh) // 2
    return r, dw, dh


def _dfl_decode(box_nchw: np.ndarray,
                H: int, W: int, stride: int) -> np.ndarray:
    """
    Distribution Focal Loss box regression decode for one feature scale.

    Corresponds to dfl_decode_side() in RKNNDetector.cpp, vectorised to
    handle all H*W anchors of one scale simultaneously.

    Args:
        box_nchw : float32  [64, H, W]  — 4 sides x REG_MAX bins, NCHW
        H, W     : feature-map spatial dimensions for this scale
        stride   : pixel stride  (8 / 16 / 32)

    Returns:
        xyxy : float32  [H*W, 4]  (x1, y1, x2, y2) in letterbox-space pixels
    """
    HW  = H * W
    # Reshape to [4 sides, REG_MAX bins, HW anchors]
    box = box_nchw.reshape(4, REG_MAX, HW).astype(np.float32)

    # Numerically-stable softmax along the REG_MAX axis (axis=1)
    bmax = box.max(axis=1, keepdims=True)
    exp  = np.exp(box - bmax)
    sfmx = exp / exp.sum(axis=1, keepdims=True)   # [4, REG_MAX, HW]

    # Weighted expectation: sum(p(b)*b), b in {0,...,REG_MAX-1}
    bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1)
    ltrb = (sfmx * bins).sum(axis=1) * stride     # [4, HW] (left,top,right,bot in px)

    # Anchor-grid cell centres in letterbox-space pixels
    xs = (np.arange(W, dtype=np.float32) + 0.5) * stride
    ys = (np.arange(H, dtype=np.float32) + 0.5) * stride
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    gx = gx.ravel()   # [HW]
    gy = gy.ravel()   # [HW]

    x1 = gx - ltrb[0]
    y1 = gy - ltrb[1]
    x2 = gx + ltrb[2]
    y2 = gy + ltrb[3]

    return np.stack([x1, y1, x2, y2], axis=1)   # [HW, 4]


def _nms(boxes: np.ndarray, scores: np.ndarray,
         iou_threshold: float) -> list:
    """Non-Maximum Suppression — pure numpy (unchanged from TFLite version)."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1  = np.maximum(x1[i], x1[rest])
        iy1  = np.maximum(y1[i], y1[rest])
        ix2  = np.minimum(x2[i], x2[rest])
        iy2  = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        iou   = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]

    return keep


# =============================================================================
# Postprocessing  (public — called from main.py)
# =============================================================================

def postprocess_detections(raw, frame, config: dict) -> list:
    """
    Decode model output into structured detections.

    Auto-detects the output format from the type of `raw`:
      raw is list    -> RKNN 9-tensor path  (airockchip RKNN-optimised export)
      raw is ndarray -> TFLite unified tensor  [1, 4+nc, 8400]

    Args:
        raw    : RKNN: list of 9 float32 ndarrays
                 TFLite: float32 ndarray [1, 4+nc, 8400]
        frame  : original BGR frame (shape used for coordinate scaling)
        config : full application config dict

    Returns:
        detections : list of {bbox:[x1,y1,x2,y2], label_id:int, confidence:float}
    """
    if isinstance(raw, list):
        return _postprocess_rknn(raw, frame, config)
    return _postprocess_tflite(raw, frame, config)


def _postprocess_rknn(outputs: list, frame, config: dict) -> list:
    """
    Decode RKNN 9-output YOLOv8 model (80-class COCO or any nc).

    outputs[s*3+0] -> box   [1, 64, Hs, Ws]  DFL distribution  NCHW
    outputs[s*3+1] -> score [1, nc, Hs, Ws]  sigmoid scores     NCHW
    outputs[s*3+2] -> sum   (ignored)

    Corresponds to RKNNDetector::postprocess() in RKNNDetector.cpp,
    extended from single-class to nc classes.

    TENSOR LAYOUT NOTE:
        The C++ debug print should show fmt=NCHW (fmt=0) for all 9 outputs.
        If your model reports fmt=NHWC (fmt=1) the DFL decode will be wrong.
        In that case transpose box/score before the loop or reshape accordingly.
    """
    global _shape_logged

    if not outputs or len(outputs) < 9:
        logger.error(
            "Expected >=9 RKNN outputs, got %d. "
            "Check model is airockchip RKNN-optimised YOLOv8 export.",
            len(outputs)
        )
        return []

    # Log output shapes once so the user can verify NCHW layout
    if not _shape_logged:
        logger.info("RKNN output tensor shapes (first inference):")
        for i, t in enumerate(outputs[:9]):
            if t is not None:
                logger.info("  output[%d]: %s  dtype=%s", i, t.shape, t.dtype)
        _shape_logged = True

    conf_thresh = config["inference"]["confidence_threshold"]
    iou_thresh  = config["inference"].get("iou_threshold", 0.45)
    max_det     = config["inference"]["max_detections"]
    fh, fw      = frame.shape[:2]

    # Letterbox parameters — must mirror preprocess_frame() exactly
    ratio, dw, dh = _lb_params(fh, fw)

    all_boxes  = []
    all_scores = []
    all_labels = []

    for s in range(N_SCALES):
        box_t   = outputs[s * 3 + 0]   # [1, 64, H, W]
        score_t = outputs[s * 3 + 1]   # [1, nc, H, W]

        if box_t is None or score_t is None:
            continue

        box_raw   = box_t[0].astype(np.float32)    # [64, H, W]
        score_raw = score_t[0].astype(np.float32)  # [nc, H, W]

        nc, H, W = score_raw.shape
        stride   = STRIDES[s]

        # Flatten spatial dims and find per-anchor max class score
        scores_flat = score_raw.reshape(nc, H * W)   # [nc, HW]
        max_scores  = scores_flat.max(axis=0)         # [HW]
        mask        = max_scores >= conf_thresh
        if not mask.any():
            continue

        label_ids = scores_flat.argmax(axis=0)            # [HW]
        xyxy_all  = _dfl_decode(box_raw, H, W, stride)    # [HW, 4] letterbox px

        # Keep only anchors above threshold
        xyxy   = xyxy_all[mask]       # [N, 4]
        confs  = max_scores[mask]     # [N]
        labels = label_ids[mask]      # [N]

        # Unproject: letterbox space -> original frame coordinates
        xyxy[:, 0::2] = (xyxy[:, 0::2] - dw) / ratio   # x1, x2
        xyxy[:, 1::2] = (xyxy[:, 1::2] - dh) / ratio   # y1, y2

        # Clamp to frame boundary
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0.0, fw)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0.0, fh)

        all_boxes.append(xyxy)
        all_scores.append(confs)
        all_labels.append(labels)

    if not all_boxes:
        return []

    boxes  = np.concatenate(all_boxes)    # [Total, 4]
    scores = np.concatenate(all_scores)   # [Total]
    labels = np.concatenate(all_labels)   # [Total]

    # Per-class NMS -> global top-k
    keep = []
    for cls in np.unique(labels):
        m   = labels == cls
        idx = np.where(m)[0]
        kept = _nms(boxes[m], scores[m], iou_thresh)
        keep.extend(idx[kept].tolist())

    keep.sort(key=lambda i: -scores[i])
    keep = keep[:max_det]

    return [
        {
            "bbox": [
                int(boxes[i, 0]), int(boxes[i, 1]),
                int(boxes[i, 2]), int(boxes[i, 3]),
            ],
            "label_id":   int(labels[i]),
            "confidence": float(scores[i]),
        }
        for i in keep
    ]


# ── TFLite legacy path  (kept so a TFLite model still works) ------------------

_buf_max_scores = None
_buf_mask       = None


def _postprocess_tflite(raw: np.ndarray, frame, config: dict) -> list:
    """
    Decode unified YOLOv8 TFLite tensor  [1, 4+nc, 8400].
    Kept for backward compatibility; not used in RKNN mode.
    """
    global _buf_max_scores, _buf_mask

    conf_thresh = config["inference"]["confidence_threshold"]
    iou_thresh  = config["inference"].get("iou_threshold", 0.45)
    max_det     = config["inference"]["max_detections"]
    fh, fw      = frame.shape[:2]

    pred   = raw[0]          # [4+nc, 8400]
    boxes  = pred[:4, :]
    scores = pred[4:, :]
    na     = scores.shape[1]

    if _buf_max_scores is None or _buf_max_scores.shape[0] != na:
        _buf_max_scores = np.empty(na, dtype=np.float32)
        _buf_mask       = np.empty(na, dtype=bool)

    np.max(scores, axis=0, out=_buf_max_scores)
    np.greater_equal(_buf_max_scores, conf_thresh, out=_buf_mask)
    mask = _buf_mask
    if not mask.any():
        return []

    confs     = _buf_max_scores[mask]
    label_ids = scores[:, mask].argmax(axis=0)
    b         = boxes[:, mask]

    cx, cy = b[0] * fw, b[1] * fh
    hw, hh = b[2] * fw / 2, b[3] * fh / 2
    px     = np.stack([cx - hw, cy - hh, cx + hw, cy + hh], axis=1)

    keep = []
    for cls in np.unique(label_ids):
        m   = label_ids == cls
        idx = np.where(m)[0]
        kept = _nms(px[m], confs[m], iou_thresh)
        keep.extend(idx[kept].tolist())

    keep.sort(key=lambda i: -confs[i])
    keep = keep[:max_det]

    x1, y1, x2, y2 = px[:, 0], px[:, 1], px[:, 2], px[:, 3]
    return [
        {
            "bbox": [
                int(np.clip(x1[i], 0, fw)), int(np.clip(y1[i], 0, fh)),
                int(np.clip(x2[i], 0, fw)), int(np.clip(y2[i], 0, fh)),
            ],
            "label_id":   int(label_ids[i]),
            "confidence": float(confs[i]),
        }
        for i in keep
    ]


# =============================================================================
# Backward-compatible wrappers  (for any code that calls these directly)
# =============================================================================

def load_model(config: dict):
    """Return (rknn_handle, input_details, output_details) for legacy callers."""
    stages = load_pipeline(config)
    s = stages[0]
    return s["rknn"], s["input_details"], s["output_details"]


def run_invoke(rknn_handle, input_details: list, output_details: list,
               input_data: np.ndarray) -> tuple:
    """Invoke RKNN model and return (outputs_list, invoke_ms)."""
    t = time.monotonic()
    outputs = rknn_handle.inference(inputs=[input_data])
    invoke_ms = (time.monotonic() - t) * 1000
    return outputs, invoke_ms


def run_inference(rknn_handle, input_details: list, output_details: list,
                  frame, config: dict) -> tuple:
    """Convenience: preprocess -> invoke -> postprocess in one call."""
    tensor    = preprocess_frame(frame, input_details)
    outputs, invoke_ms = run_invoke(rknn_handle, input_details, output_details, tensor)
    dets      = postprocess_detections(outputs, frame, config)
    return dets, invoke_ms
