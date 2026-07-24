"""
main_video_v2.py — Video dosyasından okuyarak çalışan RK3588 Edge AI Vision versiyonu.

Usage:
    python3 main_video_v2.py --video input.mp4 [--config config.json] [--loop] [--duration 300] [--no-npu] [--verbose]

main_v2.py'den farklar:
    - Kamera yerine video dosyası okunur
    - GStreamer: filesrc ! qtdemux ! h264parse ! mppvideodec (RK3588 HW decode)
    - --loop ile video sonunda başa döner
    - --duration ile belirli süre çalışır (video bitse de devam eder)
    - EOS GStreamer bus üzerinden algılanır
    - perf.stop() en sona alındı (segfault önlemi)

Performance log: perf_metrics.csv  (flush every 100 frames)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np

import actions
import inference
import overlay
import streamer
import perf_logger_v2 as perf_logger
from bytetracker_fix import BYTETracker  # FIX: bytetracker → bytetracker_fix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STATS_ALPHA   = 0.01
_PERF_LOG_PATH = 'perf_metrics.csv'
_PERF_WINDOW   = 300
_PERF_FLUSH_N  = 100

_decode_ms_shared: list[float] = [0.0]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def resolve_model_path(config: dict) -> str:
    m = config["model"]
    if "path" in m:
        return m["path"]
    source = m["variants"][m["variant"]]
    path = str(Path(m["models_dir"]) / source / "pipeline.json")
    m["path"] = path
    return path


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GStreamer video dosyası pipeline — RK3588 (mppvideodec)
# ---------------------------------------------------------------------------

def _open_gst_file_pipeline(video_path: str, input_h: int, input_w: int):
    """
    RK3588 için gi/Gst pipeline:
      filesrc ! qtdemux ! h264parse ! mppvideodec ! videoscale ! videoconvert
      ! video/x-raw,format=RGB ! appsink

    Tam çözünürlük çıkar; bgr display için kullanılır.
    Tensor resize CPU'da yapılır (videoscale'i tensor için kullanmıyoruz —
    display kalitesi kaybolmasın).

    Başarılıysa (pipeline, appsink), başarısızsa (None, None) döner.
    """
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
    except Exception as e:
        logger.info(f"GStreamer gi N/A ({e}), OpenCV fallback kullanılacak.")
        return None, None

    if Gst.ElementFactory.find('mppvideodec') is None:
        logger.info("mppvideodec bulunamadı — RK3588 değil mi? OpenCV fallback.")
        return None, None

    pipe_str = (
        f'filesrc location="{video_path}" ! '
        f'qtdemux ! h264parse ! mppvideodec ! '
        f'videoconvert ! video/x-raw,format=RGB ! '
        f'appsink name=sink sync=false max-buffers=2 drop=false'
    )
    pipeline = None
    try:
        pipeline = Gst.parse_launch(pipe_str)
        appsink  = pipeline.get_by_name('sink')
        pipeline.set_state(Gst.State.PLAYING)
        ret, _, _ = pipeline.get_state(5 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline state FAILURE")
        logger.info(f"GStreamer mppvideodec pipeline başlatıldı: {video_path}")
        return pipeline, appsink
    except Exception as e:
        logger.warning(f"GStreamer pipeline başarısız ({e}), OpenCV fallback.")
        if pipeline:
            pipeline.set_state(Gst.State.NULL)
        return None, None


# ---------------------------------------------------------------------------
# GStreamer capture loop
# ---------------------------------------------------------------------------

def _gst_file_capture_loop(video_path: str,
                            pipeline, appsink,
                            preproc_queue: Queue,
                            input_h: int, input_w: int,
                            input_dtype,
                            stop_event: threading.Event,
                            loop: bool) -> None:
    """
    RK3588 GStreamer dosya capture — main_v2.py'deki _gst_capture_loop mantığı.
    bgr: tam çözünürlük (display)
    tensor: CPU resize → NPU input boyutu
    EOS: loop=True → seek(0), loop=False → stop_event.set()
    """
    import gi
    from gi.repository import Gst

    bus = pipeline.get_bus()

    while not stop_event.is_set():
        msg = bus.timed_pop_filtered(0, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg is not None:
            if msg.type == Gst.MessageType.EOS:
                if loop:
                    logger.info("Video sonu — başa dönülüyor.")
                    pipeline.seek_simple(
                        Gst.Format.TIME,
                        Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                        0
                    )
                else:
                    logger.info("Video sonu — çıkılıyor.")
                    stop_event.set()
                    break
            elif msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error(f"GStreamer hatası: {err} | {debug}")
                stop_event.set()
                break

        t0     = time.monotonic()
        sample = appsink.emit('try-pull-sample', 100 * Gst.MSECOND)
        if sample is None:
            continue
        decode_ms = (time.monotonic() - t0) * 1000

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        s    = caps.get_structure(0)
        h, w = s.get_value('height'), s.get_value('width')

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            continue
        rgb = np.ndarray((h, w, 3), dtype=np.uint8, buffer=map_info.data).copy()
        buf.unmap(map_info)

        # Display frame — tam çözünürlük BGR
        bgr = rgb[:, :, ::-1].copy()

        # NPU tensor için CPU resize
        rgb_small = cv2.resize(rgb, (input_w, input_h)) if (h != input_h or w != input_w) else rgb

        if input_dtype == np.int8:
            tensor = np.expand_dims(rgb_small.astype(np.int16) - 128, axis=0).astype(np.int8)
        elif input_dtype == np.uint8:
            tensor = np.expand_dims(rgb_small, axis=0)
        else:
            tensor = np.expand_dims(rgb_small / 255.0, axis=0).astype(np.float32)

        _decode_ms_shared[0] = decode_ms

        if preproc_queue.full():
            try:
                preproc_queue.get_nowait()
            except Empty:
                pass
        preproc_queue.put_nowait((bgr, [tensor], 0.0))

    pipeline.set_state(Gst.State.NULL)
    logger.debug("GStreamer dosya capture thread durdu.")


# ---------------------------------------------------------------------------
# OpenCV fallback capture + preprocess
# ---------------------------------------------------------------------------

def _opencv_capture_loop(video_path: str,
                         frame_queue: Queue,
                         stop_event: threading.Event,
                         loop: bool) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"OpenCV ile video açılamadı: {video_path}")
        stop_event.set()
        return
    logger.info(f"OpenCV fallback ile video açıldı: {video_path}")

    while not stop_event.is_set():
        t0 = time.monotonic()
        ret, frame = cap.read()
        decode_ms  = (time.monotonic() - t0) * 1000

        if not ret:
            if loop:
                logger.info("Video sonu — başa dönülüyor.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                logger.info("Video sonu — çıkılıyor.")
                stop_event.set()
                break

        _decode_ms_shared[0] = decode_ms

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put_nowait(frame)

    cap.release()
    logger.debug("OpenCV capture thread durdu.")


def _preprocess_loop(frame_queue: Queue,
                     preproc_queue: Queue,
                     input_details: list,
                     stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue
        input_data = inference.preprocess_frame(frame, input_details)
        if preproc_queue.full():
            try:
                preproc_queue.get_nowait()
            except Empty:
                pass
        preproc_queue.put_nowait((frame, [input_data], 0.0))
    logger.debug("Preprocess thread durdu.")


# ---------------------------------------------------------------------------
# Stage loop
# ---------------------------------------------------------------------------

def _stage_loop(stage: dict, in_queue: Queue, out_queue: Queue,
                stop_event: threading.Event) -> None:
    label = stage["label"]
    while not stop_event.is_set():
        try:
            frame, data, invoke_ms_acc = in_queue.get(timeout=1.0)
        except Empty:
            continue
        out_data, elapsed_ms = inference.invoke_stage(stage, data)
        stage["avg_ms"] += _STATS_ALPHA * (elapsed_ms - stage["avg_ms"])
        npu_ms = elapsed_ms if stage["is_npu"] else 0.0
        if out_queue.full():
            try:
                out_queue.get_nowait()
            except Empty:
                pass
        out_queue.put_nowait((frame, out_data, invoke_ms_acc + npu_ms))
    logger.debug(f"Stage '{label}' thread durdu.")


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def run(config: dict, video_path: str, loop: bool,
        duration: float | None = None) -> None:
    stop_event = threading.Event()

    # --duration varsa video EOS'ta başa döner, süre dolunca durur
    effective_loop = loop or (duration is not None)

    # --- Pipeline ---
    logger.info("Pipeline yükleniyor...")
    pipeline = inference.load_pipeline(config)
    for stage in pipeline:
        stage["avg_ms"] = 0.0
    labels     = inference.load_labels(config["model"]["labels_path"])
    model_name = Path(config["model"]["path"]).stem
    logger.info(f"Labels: {len(labels)} | Stages: {len(pipeline)}")

    inp_det          = pipeline[0]["input_details"][0]
    input_h, input_w = inp_det['shape'][1], inp_det['shape'][2]
    input_dtype      = inp_det['dtype']

    # --- Perf logger ---
    perf = perf_logger.PerfLoggerV2(
        log_path=_PERF_LOG_PATH,
        window=_PERF_WINDOW,
        flush_every=_PERF_FLUSH_N,
    )
    logger.info(f"Perf log: {_PERF_LOG_PATH}  (flush/{_PERF_FLUSH_N} frames)")

    # --- BYTETracker ---
    track_cfg = config.get("tracking", {})
    if track_cfg.get("enabled", False):
        tracker = BYTETracker(
            frame_rate   = track_cfg.get("frame_rate",   30),
            track_buffer = track_cfg.get("track_buffer", 30),
            track_thresh = track_cfg.get("track_thresh", 0.3),
            high_thresh  = track_cfg.get("high_thresh",  0.5),
            match_thresh = track_cfg.get("match_thresh", 0.8),
        )
        logger.info(
            f"BYTETracker aktif: track_thresh={track_cfg.get('track_thresh', 0.3)} "
            f"high_thresh={track_cfg.get('high_thresh', 0.5)} "
            f"match_thresh={track_cfg.get('match_thresh', 0.8)}"
        )
    else:
        tracker = None
        logger.info("BYTETracker devre dışı.")

    # --- Capture thread'leri ---
    preproc_queue: Queue = Queue(maxsize=2)
    gst_pipe, appsink   = _open_gst_file_pipeline(video_path, input_h, input_w)

    if gst_pipe is not None:
        threading.Thread(
            target=_gst_file_capture_loop,
            args=(video_path, gst_pipe, appsink,
                  preproc_queue, input_h, input_w, input_dtype,
                  stop_event, effective_loop),
            daemon=True, name="GstFileCapture"
        ).start()
        logger.info("GStreamer mppvideodec capture başlatıldı.")
    else:
        frame_queue: Queue = Queue(maxsize=2)
        threading.Thread(
            target=_opencv_capture_loop,
            args=(video_path, frame_queue, stop_event, effective_loop),
            daemon=True, name="OpenCVCapture"
        ).start()
        threading.Thread(
            target=_preprocess_loop,
            args=(frame_queue, preproc_queue, pipeline[0]["input_details"], stop_event),
            daemon=True, name="Preprocess"
        ).start()
        logger.info("OpenCV fallback capture başlatıldı.")

    # --- NPU stage thread'leri ---
    stage_in_queue = preproc_queue
    for stage in pipeline:
        stage_out_queue: Queue = Queue(maxsize=2)
        threading.Thread(
            target=_stage_loop,
            args=(stage, stage_in_queue, stage_out_queue, stop_event),
            daemon=True, name=f"Stage-{stage['label']}"
        ).start()
        logger.info(f"Stage '{stage['label']}' başlatıldı.")
        stage_in_queue = stage_out_queue
    raw_queue = stage_in_queue

    # --- Streamer ---
    streamer.start(config)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        board_ip = s.getsockname()[0]
        s.close()
    except Exception:
        board_ip = "0.0.0.0"
    stream_url = f"http://{board_ip}:{config['streaming']['port']}"
    logger.info(f"Stream : {stream_url}")
    logger.info(f"Status : {stream_url}/status")

    # --- EMA ---
    fps_ema          = 0.0
    invoke_ms_avg    = 0.0
    inference_ms_avg = 0.0
    frame_count      = 0
    t_start          = time.monotonic()
    t_prev_frame: float | None = None

    logger.info("Inference döngüsü başladı. Durdurmak için Ctrl+C.")

    try:
        while not stop_event.is_set():
            try:
                frame, raw_list, invoke_ms = raw_queue.get(timeout=1.0)
            except Empty:
                if stop_event.is_set():
                    break
                logger.warning("1s içinde frame gelmedi — bekleniyor...")
                continue

            t_got    = time.monotonic()
            frame_ms = (t_got - t_prev_frame) * 1000.0 if t_prev_frame else 33.3
            t_prev_frame = t_got

            # Duration kontrolü
            if duration is not None and (t_got - t_start) >= duration:
                logger.info(f"Hedef süreye ulaşıldı ({duration:.0f}s) — durduruluyor.")
                stop_event.set()
                break

            decode_ms = _decode_ms_shared[0]

            # Dequantize — TFLite int8 only
            raw = raw_list[0]
            if isinstance(raw, np.ndarray) and raw.dtype == np.int8:
                last_out_det = pipeline[-1]["output_details"]
                scale, zero_point = last_out_det[0]["quantization"]
                raw = (raw.astype(np.float32) - zero_point) * scale

            # Postprocess
            t_post = time.monotonic()
            if config["inference"].get("skip_postprocess", False):
                detections = []
            else:
                detections = inference.postprocess_detections(raw, frame, config)
            postproc_ms = (time.monotonic() - t_post) * 1000

            # BYTETracker
            tracks = tracker.update(detections) if tracker else None

            # Perf
            perf.update(
                frame_ms    = frame_ms,
                decode_ms   = decode_ms,
                invoke_ms   = invoke_ms,
                postproc_ms = postproc_ms,
            )

            # Annotate + stream
            annotated = overlay.annotate_frame(
                frame, detections, labels, fps_ema, config,
                model_name, postproc_ms, invoke_ms,
                tracks=tracks,
            )
            actions.on_frame(frame, detections, labels, config)

            stage_latency = [
                {"label": s["label"], "avg_ms": round(s["avg_ms"], 1)}
                for s in pipeline
            ]
            streamer.push_frame(
                annotated, detections, labels, fps_ema,
                invoke_ms=invoke_ms_avg, inference_ms=inference_ms_avg,
                stage_latency=stage_latency,
            )

            instant_fps       = 1000.0 / frame_ms if frame_ms > 0 else 0.0
            fps_ema          += _STATS_ALPHA * (instant_fps   - fps_ema)
            invoke_ms_avg    += _STATS_ALPHA * (invoke_ms     - invoke_ms_avg)
            inference_ms_avg += _STATS_ALPHA * (postproc_ms   - inference_ms_avg)

            frame_count += 1
            if frame_count % 100 == 0:
                uptime = time.monotonic() - t_start
                st = {m: perf._windows[m].stats() for m in perf_logger.METRICS}
                logger.info(
                    f"[{frame_count:>5}] "
                    f"FPS avg={st['fps']['avg']:.1f} min={st['fps']['min']:.1f} max={st['fps']['max']:.1f} | "
                    f"frame_ms avg={st['frame_ms']['avg']:.1f} | "
                    f"invoke avg={st['invoke_ms']['avg']:.1f}ms | "
                    f"CPU={st['cpu_pct']['avg']:.0f}% | "
                    f"NPU(C0,1,2)=({st['npu_core0']['avg']:.0f}%,{st['npu_core1']['avg']:.0f}%,{st['npu_core2']['avg']:.0f}%) | "
                    f"RAM={st['ram_mb']['avg']:.0f}MB | "
                    f"PWR={st['power_w']['avg']:.2f}W | "
                    f"up={uptime:.0f}s"
                )

    except KeyboardInterrupt:
        logger.info("Kullanıcı durdurdu.")
    finally:
        stop_event.set()
        perf.flush()
        logger.info("\n" + perf.summary())
        logger.info(f"Metrics → {_PERF_LOG_PATH}")
        if tracker is not None:
            logger.info(tracker.summary())
        logger.info("Stopped. Goodbye.")
        perf.stop()   # NPUMonitor thread'i en son durdur (segfault önlemi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RK3588 Edge AI Vision — Video dosyası modu"
    )
    parser.add_argument("--video",    required=True,       help="İşlenecek video dosyası (.mp4, .avi, ...)")
    parser.add_argument("--config",   default="config.json")
    parser.add_argument("--loop",     action="store_true", help="Video bitince başa dön")
    parser.add_argument("--duration", type=float, default=None,
                        help="Kaç saniye çalışsın (video bitse de devam eder)")
    parser.add_argument("--no-npu",   action="store_true")
    parser.add_argument("--verbose",  "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video dosyası bulunamadı: {args.video}")
        sys.exit(1)

    config = load_config(args.config)
    resolve_model_path(config)
    logger.info(f"Model  : {config['model']['path']}")
    logger.info(f"Video  : {video_path}  loop={args.loop}"
                + (f"  duration={args.duration}s" if args.duration else ""))

    if args.no_npu:
        config["model"]["use_npu"] = False
        logger.info("NPU --no-npu ile devre dışı bırakıldı.")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run(config, str(video_path), loop=args.loop, duration=args.duration)


if __name__ == "__main__":
    main()
