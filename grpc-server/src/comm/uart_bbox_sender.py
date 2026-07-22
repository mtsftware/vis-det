"""UartBboxSender -- Faz 2 host uygulamasi.

CLAUDE.md PHASE 2 SS3: kullanicinin GUI'de cizdigi BBOX'u board'a
/dev/ttyUSB0 uzerinden MAVLINK BBOX mesaji olarak gonderir.

Referans: host-app/src/mavlink/mavlink.py (tracking-app-python'dan kopyalandi),
          tracking-app/include/comm/UartMavlinkLink.hpp (board tarafi karsiligi)
"""
from __future__ import annotations

from typing import Optional

import serial

from src.mavlink.mavlink import MAVLink


class UartBboxSender:
    def __init__(self, device: str = "/dev/ttyUSB0", baud: int = 57600) -> None:
        self._device = device
        self._baud = baud
        self._serial: Optional[serial.Serial] = None
        self._mav: Optional[MAVLink] = None

    def start(self) -> bool:
        try:
            self._serial = serial.Serial(self._device, self._baud, timeout=0)
        except serial.SerialException as exc:
            print(f"[UartBboxSender] {self._device} acilamadi: {exc}")
            return False

        self._mav = MAVLink(self._serial, srcSystem=1, srcComponent=1)
        print(f"[UartBboxSender] {self._device} acildi (baud={self._baud})")
        return True

    def stop(self) -> None:
        if self._serial is not None:
            self._serial.close()
        self._serial = None
        self._mav = None

    def send_bbox(self, x: float, y: float, w: float, h: float) -> None:
        """x,y,w,h: GUI'de cizilen dikdortgen (piksel, sol-ust + boyut)."""
        if self._mav is None:
            print("[UartBboxSender] gonderilemedi: baglanti acik degil")
            return
        self._mav.bbox_send(x, y, w, h)
        print(f"[UartBboxSender] BBOX gonderildi: ({x:.0f},{y:.0f},{w:.0f},{h:.0f})")
