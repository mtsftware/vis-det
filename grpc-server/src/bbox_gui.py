"""BboxGui -- Faz 2 host uygulamasi.

CLAUDE.md PHASE 2 SS2: 1280x720 varsayilan pencerede board'un RTSP akisini
gosterir; kullanici fareyle surukleyerek bir bounding box cizer. Kare,
gosterilmeden once pencere boyutuna (varsayilan 1280x720) resize edilir --
bu yuzden fare koordinatlari, board'a gonderilen BBOX koordinatlariyla
DOGRUDAN ayni piksel uzayindadir (board'un RTSP cikisi da varsayilan
1280x720, bkz. tracking-app/src/pipeline/main_m10_test.cpp out_w/out_h).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2

BboxSelectedCallback = Callable[[float, float, float, float], None]


class BboxGui:
    def __init__(
        self,
        window_name: str = "tracking-app-host",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self._window_name = window_name
        self._width = width
        self._height = height

        self._drawing = False
        self._start: Optional[Tuple[int, int]] = None
        self._current_bbox: Optional[Tuple[int, int, int, int]] = None

        self._on_bbox_selected: Optional[BboxSelectedCallback] = None
        self._latest_target: Optional[Tuple[float, float]] = None

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, self._width, self._height)
        cv2.setMouseCallback(self._window_name, self._on_mouse)

    def set_bbox_selected_callback(self, cb: BboxSelectedCallback) -> None:
        self._on_bbox_selected = cb

    def set_target_point(self, x: float, y: float) -> None:
        self._latest_target = (x, y)

    def show(self, frame) -> int:
        """Kareyi (BGR, numpy array) goster; surukleme onizlemesi ve son
        Target Point overlay'ini ciz. cv2.waitKey(1) sonucunu dondurur."""
        display = cv2.resize(frame, (self._width, self._height))

        if self._current_bbox is not None:
            x, y, w, h = self._current_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self._latest_target is not None:
            tx, ty = self._latest_target
            cv2.drawMarker(
                display,
                (int(tx), int(ty)),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            cv2.putText(
                display,
                f"target: ({tx:.0f}, {ty:.0f})",
                (int(tx) + 12, int(ty) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow(self._window_name, display)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(self._window_name)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        x = max(0, min(x, self._width - 1))
        y = max(0, min(y, self._height - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._current_bbox = (x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            sx, sy = self._start
            self._current_bbox = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            sx, sy = self._start
            bx, by, bw, bh = min(sx, x), min(sy, y), abs(x - sx), abs(y - sy)
            self._current_bbox = (bx, by, bw, bh)
            if bw > 2 and bh > 2 and self._on_bbox_selected:
                self._on_bbox_selected(float(bx), float(by), float(bw), float(bh))
