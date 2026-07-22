"""M11: Host (GCS) tarafi TCP/gRPC test/simulasyon uygulamasi.

CLAUDE.md "PHASE 2: Host Server Application" karsiligi, TCP/gRPC ile:
  1. RTSP Monitor         -- board'un RTSP cikisini izler (app/main.py ile ayni).
  2. GUI & BBox Selection -- fare ile bbox cizimi (app/main.py ile ayni GUI).
  3. gRPC Server          -- board'un (client) Stream RPC'si ile abone
                              olacagi servisi host:port'ta acar; cizilen
                              BBox'i board'a YAZAR.

Target Point telemetrisi (board -> GCS) SIMDILIK KALDIRILDI -- akis artik
tek yonlu (GCS -> board), bidirectional stream yok. Roller M10/Faz-2'ye
gore TERSTIR: board artik gRPC CLIENT, bu uygulama gRPC SERVER (bkz.
tracking-app/include/comm/GrpcBboxSubscriber.hpp).

Bu uygulama BOARD'DAN AYRI bir makinede (GCS/yer istasyonu) calisir --
proto/tracking.proto dosyasi bu yuzden tracking-app'e degil, mavlink-server
icine (proto/tracking.proto) kendi kopyasi olarak konmustur (bkz. o dosyanin
basindaki "ONEMLI" notu -- board tarafiyla elle senkron tutulmali).

CALISTIRMADAN ONCE (bu ortamda protoc CALISTIRILMADI -- CLAUDE.md kisiti
geregi build/test adimlari board/host uzerinde ayrica yapilmali):
  cd mavlink-server
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python3 -m grpc_tools.protoc \
    -I proto \
    --python_out=generated --grpc_python_out=generated \
    proto/tracking.proto

Calistirma (proje kokunden -- mavlink-server/, app/ ve src/ kardes dizinler):
  python3 app/grpc_main.py rtsp://<board_ip>:8557/out --port 50051

Board tarafinda karsilik gelen calistirma:
  ./m11_pipeline_test rtsp://<board_ip>:8554/stream --gcs-host <bu_makinenin_ip'si> --gcs-port 50051
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# app/grpc_main.py dogrudan "python3 app/grpc_main.py" ile calistirildiginda
# Python sys.path[0]'a SADECE app/ dizinini ekler -- "from src...." (proje
# kokunun kardesi) ve "import tracking_pb2*" (generated/ dizini) importlari
# calisma dizininden bagimsiz calissin diye ikisini de elle path'e ekliyoruz.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))

import cv2  # noqa: E402

from src.comm.grpc_bbox_server import GrpcBboxServer  # noqa: E402
from src.gui.bbox_gui import BboxGui  # noqa: E402
from src.gui.latest_frame_capture import LatestFrameCapture  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tracking host uygulamasi -- M11 TCP/gRPC")
    p.add_argument("rtsp_url", help="Board'un RTSP cikisi, orn. rtsp://<board_ip>:8557/out")
    p.add_argument("--host", default="0.0.0.0", help="gRPC server bind adresi")
    p.add_argument(
        "--port", type=int, default=50051,
        help="gRPC server portu (board --gcs-port ile eslesmeli)",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # LatestFrameCapture: cv2.VideoCapture'ın ham kullanımı, okuma döngüsü
    # kaynaktan bir miktar bile yavaş kalırsa iç kuyrukta kare biriktirip
    # zamanla katlanarak artan bir gecikmeye yol açıyordu (QGC'nin aynı
    # RTSP akışını gecikmesiz göstermesiyle karşılaştırınca ortaya çıktı).
    # Bu sarmalayıcı arka planda sürekli okuyup sadece en son kareyi tutar.
    cap = LatestFrameCapture(args.rtsp_url)
    if not cap.isOpened():
        print(f"[Host] RTSP akisi acilamadi: {args.rtsp_url}")
        return 1
    cap.start()

    server = GrpcBboxServer(args.host, args.port)
    if not server.start():
        print("[Host] gRPC server baslatilamadi")
        cap.release()
        return 1

    gui = BboxGui(width=args.width, height=args.height)

    def on_bbox_selected(x: float, y: float, w: float, h: float) -> None:
        server.send_bbox(x, y, w, h)

    gui.set_bbox_selected_callback(on_bbox_selected)

    print(
        f"[Host] gRPC {args.host}:{args.port} dinleniyor -- bbox cizmek icin "
        "surukleyin, cikmak icin 'q'"
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Henuz ilk kare gelmedi (LatestFrameCapture arka planda
                # okuyor) -- busy-loop'a girmemek icin kisa bir bekleme.
                time.sleep(0.05)
                continue

            key = gui.show(frame)
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        gui.close()
        cap.release()
        server.stop()
        print("[Host] Durduruldu.")

    return 0


if __name__ == "__main__":
    sys.exit(main())