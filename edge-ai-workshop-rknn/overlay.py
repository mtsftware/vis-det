"""
overlay.py — OpenCV drawing utilities for detection and tracking visualization.

Track ID'li çizim için draw_all_tracks() ve annotate_frame(tracks=...) eklendi.
"""


from __future__ import annotations

import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 165, 0),  # orange
    (128, 0, 128),  # purple
    (0, 128, 128),  # teal
]

# Track ID'ye göre renk — geniş palet, her track tutarlı renk alır
_TRACK_COLORS = [
    (255, 56,  56),   (255, 157, 151),  (255, 112, 31),   (255, 178, 29),
    (207, 210,  49),  (72,  249,  10),  (146, 204,  23),  (61,  219, 134),
    (26,  147, 52),   (0,   212, 187),  (44,  153, 168),  (0,   194, 255),
    (52,  69,  147),  (100, 115, 255),  (0,   24,  236),  (132,  56, 255),
    (82,   0, 133),   (203,  56, 255),  (255, 149, 200),  (255,  55, 199),
]


def get_color_for_label(label_id: int) -> tuple:
    return COLORS[label_id % len(COLORS)]


def get_color_for_track(track_id: int) -> tuple:
    """Track ID'ye kararlı renk döndür (her seferinde aynı renk)."""
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


# ---------------------------------------------------------------------------
# Dedektör çizimi (orijinal, değişmedi)
# ---------------------------------------------------------------------------

def draw_detection(frame, detection: dict, label: str, config: dict) -> None:
    """Tek bir dedektör kutusunu çiz."""
    x1, y1, x2, y2 = detection["bbox"]
    confidence = detection["confidence"]
    label_id   = detection["label_id"]

    thickness       = config["overlay"].get("box_thickness", 2)
    font_scale      = config["overlay"].get("font_scale", 0.6)
    show_confidence = config["overlay"].get("show_confidence", True)

    color = get_color_for_label(label_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = f"{label}: {confidence:.0%}" if show_confidence else label
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    ly = max(y1 - 5, th + 5)
    cv2.rectangle(frame, (x1, ly - th - bl), (x1 + tw, ly + bl), color, cv2.FILLED)
    cv2.putText(frame, text, (x1, ly - bl // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_all_detections(frame, detections: list[dict],
                        labels: list[str], config: dict) -> None:
    """Tüm dedektörleri çiz (track olmadan)."""
    for det in detections:
        lid   = det["label_id"]
        label = labels[lid] if lid < len(labels) else str(lid)
        draw_detection(frame, det, label, config)


# ---------------------------------------------------------------------------
# Track çizimi (YENİ)
# ---------------------------------------------------------------------------

def draw_track(frame, track: dict, labels: list[str], config: dict) -> None:
    """
    Tek bir track'i çiz.

    track sözlüğü: {'bbox', 'label_id', 'confidence', 'track_id'}
    """
    x1, y1, x2, y2 = track["bbox"]
    confidence = track["confidence"]
    label_id   = track["label_id"]
    track_id   = track["track_id"]

    thickness       = config["overlay"].get("box_thickness", 2)
    font_scale      = config["overlay"].get("font_scale", 0.6)
    show_confidence = config["overlay"].get("show_confidence", True)

    label = labels[label_id] if label_id < len(labels) else str(label_id)
    color = get_color_for_track(track_id)

    # Kutu
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Track ID etiketi — köşe üstünde küçük rozet
    id_text = f"#{track_id}"
    (iw, ih), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - ih - 6), (x1 + iw + 4, y1), color, cv2.FILLED)
    cv2.putText(frame, id_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Label + güven etiketi — kutunun altında
    if show_confidence:
        det_text = f"{label}: {confidence:.0%}"
    else:
        det_text = label
    (tw, th), bl = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    ty = min(y2 + th + 8, frame.shape[0] - 2)
    cv2.rectangle(frame, (x1, ty - th - bl - 2), (x1 + tw + 2, ty + 2), color, cv2.FILLED)
    cv2.putText(frame, det_text, (x1 + 1, ty - bl),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_all_tracks(frame, tracks: list[dict],
                    labels: list[str], config: dict) -> None:
    """Tüm track'leri track ID rozetleriyle çiz."""
    for t in tracks:
        draw_track(frame, t, labels, config)


# ---------------------------------------------------------------------------
# ROI zone (değişmedi)
# ---------------------------------------------------------------------------

def draw_roi_zone(frame, zone: tuple,
                  color=(0, 200, 255), label: str = "Zone") -> None:
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD (değişmedi)
# ---------------------------------------------------------------------------

def draw_hud(frame, fps: float, num_detections: int, config: dict,
             model_name: str = "", inference_ms: float = 0.0,
             invoke_ms: float = 0.0) -> None:
    if not config["overlay"].get("show_fps", True):
        return

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color      = (255, 255, 255)
    shadow     = (0, 0, 0)
    thickness  = 1

    lines = [
        f"FPS: {fps:.1f}",
        f"Invoke: {invoke_ms:.1f} ms  Post: {inference_ms:.1f} ms",
        f"Detections: {num_detections}",
    ]
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append(datetime.now().strftime("%H:%M:%S"))

    y = 20
    for line in lines:
        cv2.putText(frame, line, (11, y + 1), font, font_scale, shadow, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),     font, font_scale, color,  thickness,     cv2.LINE_AA)
        y += 22


# ---------------------------------------------------------------------------
# annotate_frame  ← tracks parametresi eklendi, geriye uyumlu
# ---------------------------------------------------------------------------

def annotate_frame(frame, detections: list[dict], labels: list[str],
                   fps: float, config: dict, model_name: str = "",
                   inference_ms: float = 0.0, invoke_ms: float = 0.0,
                   tracks: list[dict] | None = None) -> np.ndarray:
    """
    Frame'i kopyala, annotation ekle ve döndür.

    tracks verilirse track ID rozetleriyle çizim yapılır.
    tracks=None ise ham dedektör kutuları çizilir (önceki davranış).
    """
    annotated = frame.copy()

    if tracks is not None:
        # BYTETracker çıktısı — Kalman tahminli, track ID'li kutular
        draw_all_tracks(annotated, tracks, labels, config)
        n_shown = len(tracks)
    else:
        # Orijinal ham dedektör kutuları
        draw_all_detections(annotated, detections, labels, config)
        n_shown = len(detections)

    draw_hud(annotated, fps, n_shown, config, model_name, inference_ms, invoke_ms)

    return annotated
