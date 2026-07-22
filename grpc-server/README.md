# mavlink-server (host / GCS uygulaması)

Board'dan (RK3588, `tracking-app/`) **ayrı bir makinede** çalışan yer
istasyonu (GCS) uygulaması. İki bağımsız varyant içerir:

- **`app/grpc_main.py`** (M11, güncel) — TCP/gRPC üzerinden BBOX gönderimi
  + Target Point alımı. Bu makine gRPC **server**, board gRPC **client**.
- **`app/main.py`** (Faz 2, referans) — UART/MAVLINK + UDP/MAVLINK üzerinden
  aynı işlev. M10 ile eşleşir, M11 için karşılaştırma amaçlı korunuyor.

Bu dizin board reposundan (`tracking-app/`) tamamen bağımsız çalışacak
şekilde kendi bağımlılıklarını (`requirements.txt`) ve proto kopyasını
(`proto/tracking.proto`) barındırır — ortak bir dosya sistemi paylaşmazlar.

## Kurulum (bu makinede, GCS simülasyonu için)

```bash
cd mavlink-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Proto kodunu üret

```bash
python3 -m grpc_tools.protoc \
  -I proto \
  --python_out=generated --grpc_python_out=generated \
  proto/tracking.proto
ls generated/   # tracking_pb2.py, tracking_pb2_grpc.py görünmeli
```

`proto/tracking.proto`, board'daki `tracking-app/proto/tracking.proto` ile
**elle senkron tutulan bir kopyadır** (ayrı makine/repo oldukları için tek
kaynaktan üretilmiyor). Servis/mesaj tanımını değiştirirsen ikisini de
güncelle — aksi halde iki taraf farklı proto'dan üretilmiş kodla konuşur ve
stream açılsa bile alanlar yanlış çözümlenir.

## Çalıştırma

```bash
python3 app/grpc_main.py rtsp://<board_ip>:8557/out --port 50051
```

Board tarafında (ayrı terminalde, `tracking-app/` içinde):

```bash
./m11_pipeline_test rtsp://<kaynak_ip>:8554/stream \
  --gcs-host <bu_makinenin_ip'si> --gcs-port 50051
```

GUI'de fareyle bbox çiz → gRPC ile board'a gider → board tracker'ı
etkinleştirir → target point gRPC ile geri gelir → GUI'de kırmızı çarpı
olarak görünür.

## Dizin yapısı

```
mavlink-server/
├── proto/tracking.proto        # board ile senkron tutulan proto kopyası
├── generated/                  # protoc çıktısı (commit edilmez, .gitignore)
├── requirements.txt
├── app/
│   ├── grpc_main.py             # M11 giriş noktası (gRPC)
│   └── main.py                  # Faz 2 giriş noktası (MAVLink, referans)
└── src/
    ├── comm/grpc_tracking_server.py   # M11 gRPC server sınıfı
    ├── comm/uart_bbox_sender.py       # Faz 2 referans
    ├── comm/udp_target_point_listener.py  # Faz 2 referans
    ├── gui/bbox_gui.py               # RTSP izleme + bbox çizim GUI (ortak)
    └── mavlink/mavlink.py            # Faz 2 referans (MAVLink kod üretimi)
```
