"""GrpcBboxServer -- M11 host (GCS) tarafi, TCP/gRPC uzerinden (tek yonlu:
GCS -> board).

CLAUDE.md "PHASE 2: Host Server Application" karsiligi: board (client) bu
server'a baglanir, Stream (server-streaming) RPC'si ile abone olur; GUI'de
cizilen her BBox stream uzerinden board'a YAZILIR. TargetPoint alimi (eski
UdpTargetPointListener) ve bidirectional stream SIMDILIK KALDIRILDI --
akis artik tek yonlu.

SendBoundingBox unary RPC'si, harici bir uretici (test araci, otomasyon
betigi) tek bir BBox'i dogrudan aga push edip aktif Stream abonelerine
yayinlatabilsin diye ayrica sunuluyor -- GCS'in kendi GUI'si send_bbox()
ile ayni ic yayin mekanizmasini dogrudan kullanir, aga cikmaz.

Referans: proto/tracking.proto (mesaj/servis tanimi -- board'daki
          tracking-app/proto/tracking.proto ile elle senkron tutulan kopya),
          tracking-app/include/comm/GrpcBboxSubscriber.hpp (board tarafi
          karsiligi, ayri repo/makine)
"""
from __future__ import annotations

import queue
import threading
from concurrent import futures
from typing import List, Optional

import grpc

# protoc --grpc_python_out ile uretilen tracking_pb2_grpc.py, tracking_pb2'yi
# "import tracking_pb2" (paket-nispi DEGIL, duz) seklinde import eder --
# bu yuzden "generated" bir paket olarak degil, dogrudan sys.path'e eklenen
# bir dizin olarak kullaniliyor (bkz. app/grpc_main.py path kurulumu).
import tracking_pb2
import tracking_pb2_grpc


class _DroneObjectTrackingServicer(tracking_pb2_grpc.DroneObjectTrackingServiceServicer):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue[tracking_pb2.BBox]"] = []

    def broadcast(self, bbox: tracking_pb2.BBox) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put(bbox)

    def SendBoundingBox(self, request, context):
        self.broadcast(request)
        print(
            f"[GrpcBboxServer] BBox alindi (SendBoundingBox): "
            f"({request.x:.0f},{request.y:.0f},{request.w:.0f},{request.h:.0f})"
        )
        return tracking_pb2.BoundingBoxAck(success=True, message="ok")

    def Stream(self, request, context):
        q: "queue.Queue[tracking_pb2.BBox]" = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        print("[GrpcBboxServer] board abone oldu (Stream)")

        try:
            while context.is_active():
                try:
                    bbox = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                yield bbox
        finally:
            with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)
            print("[GrpcBboxServer] board abonelikten cikti")


class GrpcBboxServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 50051) -> None:
        self._host = host
        self._port = port
        self._server: Optional[grpc.Server] = None
        self._servicer = _DroneObjectTrackingServicer()

    def start(self) -> bool:
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        tracking_pb2_grpc.add_DroneObjectTrackingServiceServicer_to_server(
            self._servicer, self._server
        )

        addr = f"{self._host}:{self._port}"
        bound_port = self._server.add_insecure_port(addr)
        if bound_port == 0:
            print(f"[GrpcBboxServer] {addr} baglanamadi")
            self._server = None
            return False

        self._server.start()
        print(f"[GrpcBboxServer] {addr} dinleniyor")
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(grace=1.0)
        self._server = None

    def send_bbox(self, x: float, y: float, w: float, h: float) -> None:
        """x,y,w,h: GUI'de cizilen dikdortgen (piksel, sol-ust + boyut).
        Aktif tum Stream abonelerine (board'lara) yayinlanir -- su an bagli
        bir board olmasa bile yayin denemesi sessizce hicbir seye ulasmaz
        (kuyruk yok, abone kalmayana kadar bekleyen bir mesaj biriktirme
        YOK, M10'daki UdpTargetPointSender'in best-effort davranisiyla ayni
        ruh)."""
        bbox = tracking_pb2.BBox(x=x, y=y, w=w, h=h)
        self._servicer.broadcast(bbox)
        print(
            f"[GrpcBboxServer] BBox yayinlandi: ({x:.0f},{y:.0f},{w:.0f},{h:.0f})"
        )