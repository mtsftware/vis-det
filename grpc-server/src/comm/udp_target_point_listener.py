"""UdpTargetPointListener -- Faz 2 host uygulamasi.

CLAUDE.md PHASE 2 SS4: board'dan UDP ile gelen MAVLINK TARGET_POINT
mesajlarini dinler, en son (x, y) konumunu tutar ve bir callback ile
haber verir (terminale yazdirma + GUI overlay'i cagiran tarafin isi).

Referans: host-app/src/mavlink/mavlink.py (tracking-app-python'dan kopyalandi),
          tracking-app/include/comm/UdpTargetPointSender.hpp (board tarafi karsiligi)
"""
from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from src.mavlink.mavlink import MAVLink, MAVLink_target_point_message


@dataclass
class TargetPoint:
    x: float
    y: float


class _NoWriteFile:
    """MAVLink sinifi kurucuda bir 'file' nesnesi bekler (send() icin
    kullanilir) -- bu dinleyici asla gondermez, sadece parse_buffer()
    cagirir, o yuzden gercek bir soket/dosya gerekmiyor."""

    def write(self, _data: bytes) -> None:
        raise NotImplementedError(
            "UdpTargetPointListener sadece alici; MAVLink.send() cagrilmamali"
        )


class UdpTargetPointListener:
    def __init__(self, port: int = 14550) -> None:
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._mav = MAVLink(_NoWriteFile())

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._latest: Optional[TargetPoint] = None
        self._on_target: Optional[Callable[[TargetPoint], None]] = None

    def set_target_callback(self, cb: Callable[[TargetPoint], None]) -> None:
        self._on_target = cb

    def start(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("0.0.0.0", self._port))
            self._sock.settimeout(0.2)
        except OSError as exc:
            print(f"[UdpTargetPointListener] port {self._port} acilamadi: {exc}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[UdpTargetPointListener] UDP {self._port} dinleniyor")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
        self._sock = None

    def latest(self) -> Optional[TargetPoint]:
        with self._lock:
            return self._latest

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue

            # Her UDP datagrami tam olarak bir MAVLINK mesaji tasir (board
            # tarafi UdpTargetPointSender::send() tek bir sendto() cagrisinda
            # tek mesaj gonderiyor) -- parse_buffer bu tek paketi coz(er).
            msgs = self._mav.parse_buffer(data) or []
            for msg in msgs:
                if isinstance(msg, MAVLink_target_point_message):
                    tp = TargetPoint(x=msg.x, y=msg.y)
                    with self._lock:
                        self._latest = tp
                    if self._on_target:
                        self._on_target(tp)
