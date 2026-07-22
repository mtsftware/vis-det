"""LatestFrameCapture -- RTSP dusuk gecikme icin arka planda surekli okuyup
sadece EN SON kareyi tutan bir cv2.VideoCapture sarmalayicisi.

cv2.VideoCapture (FFmpeg backend) kendi ic kuyruguna kare biriktirir --
okuma dongusu (cap.read() + goruntu isleme) kaynaktan bir miktar bile
yavas kalsa kuyruk surekli buyur ve gecikme zamanla katlanarak artar (QGC
gibi GStreamer tabanli RTSP izleyicilerde bu sorun yok, cunku onlar
surekli en son kareyi gosterip eskileri atar -- hicbir kuyruk birikmez).
Bu sinif ayni davranisi tekrar eder: arka plan thread'i cap.read()'i
durmadan cagirir, her seferinde SADECE en son kareyi saklar -- ana dongu
ne zaman isteseydi de en guncel kareyi alir, arada uretilmis eski kareler
sessizce atilir.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


class LatestFrameCapture:
    def __init__(self, rtsp_url: str) -> None:
        self._cap = cv2.VideoCapture(rtsp_url)
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """En son kareyi dondurur. Henuz hic kare gelmediyse (ok=False)
        dondurur -- ayni kareyi tekrar okumak (henuz yenisi gelmediyse)
        sorun degildir, cagiran taraf kendi hizinda (orn. cv2.waitKey(1))
        calisir, gec kalinmis eski kareler asla BIRIKMEZ."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()