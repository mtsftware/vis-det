"""
main.py — Entry point for the FRDM-IMX95 Edge AI Vision application.

Usage:
    python3 main.py [--config config.json] [--no-npu] [--verbose]

Performance log: /tmp/perf_metrics.csv  (flush every 100 frames)
"""


from __future__ import annotations

import argparse
import json
import logging
import random
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
import perf_logger
from bytetracker import BYTETracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STATS_ALPHA   = 0.01
_PERF_LOG_PATH = 'perf_metrics.csv'
_PERF_WINDOW   = 300
_PERF_FLUSH_N  = 100

# ---------------------------------------------------------------------------
# Shared decode timing
# decode_ms, yazılan anlık değer — capture thread yazar, main loop okur.
# Pipeline pipelined olduğu için mükemmel senkronizasyon gerekmez.
# ---------------------------------------------------------------------------
_decode_ms_shared: list[float] = [0.0]


# ---------------------------------------------------------------------------
# Camera enumeration
# ---------------------------------------------------------------------------

def _list_cameras() -> list[dict]:
    import subprocess
    cameras: list[dict] = []
    seen_names: set[str] = set()
    try:
        out = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=3
        ).stdout
        current_name: str | None = None
        is_usb: bool = False
        for line in out.splitlines():
            stripped = line.strip()
            if not stripped:
                current_name, is_usb = None, False
            elif not line[0].isspace():
                current_name = stripped.rstrip(':')
                is_usb = 'usb' in current_name.lower()
            elif (is_usb and stripped.startswith('/dev/video')
                  and current_name not in seen_names):
                seen_names.add(current_name)
                cameras.append({'device': stripped, 'name': current_name})
    except Exception:
        pass
    return cameras


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
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def set_auto_exposure_on_usb_cameras(device: str) -> None:
    import subprocess
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl=auto_exposure=1"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=2
        )
        logger.info(f"Set auto_exposure=1 on {device}")
    except Exception as e:
        logger.error(f"Could not set auto exposure on {device}: {e}")


# ---------------------------------------------------------------------------
# Capture / preprocess threads
# ---------------------------------------------------------------------------

def _open_capture(cam_cfg: dict) -> tuple[cv2.VideoCapture, str]:
    url = cam_cfg.get("remote_url", "http://10.42.0.1:5001/stream")
    logger.info(f"Remote camera — connecting to {url} via GStreamer")
    gst_pipeline = (
        f'souphttpsrc location="{url}" is-live=true ! '
        f'multipartdemux ! jpegdec ! videoconvert ! '
        f'appsink sync=false max-buffers=2 drop=true'
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logger.error(f"GStreamer could not open: {url}")
        sys.exit(1)
    logger.info("GStreamer pipeline connected.")
    return cap, f"remote_gst:{url}"


def _capture_loop(cap: cv2.VideoCapture, frame_queue: Queue,
                  stop_event: threading.Event) -> None:
    """Capture frames; time each read for decode_ms."""
    while not stop_event.is_set():
        t0 = time.monotonic()
        ret, frame = cap.read()
        decode_ms = (time.monotonic() - t0) * 1000

        if not ret:
            logger.warning("Frame read failed — retrying...")
            time.sleep(0.05)
            continue

        _decode_ms_shared[0] = decode_ms

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put_nowait(frame)
    cap.release()
    logger.debug("Capture thread stopped.")


def _preprocess_loop(frame_queue: Queue, preproc_queue: Queue,
                     input_details: list, stop_event: threading.Event) -> None:
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
    logger.debug("Preprocess thread stopped.")


def _open_gst_pipeline(cam_cfg: dict):
    if not cam_cfg.get("use_gstreamer", False):
        return None, None
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
    except Exception as e:
        logger.info(f"GStreamer N/A ({e}), using OpenCV.")
        return None, None

    if Gst.ElementFactory.find('imxvideoconvert_g2d') is None:
        logger.info("imxvideoconvert_g2d not found, using OpenCV.")
        return None, None

    source = cam_cfg.get("source", "local")
    if source == "remote":
        url = cam_cfg.get("remote_url", "http://192.168.7.1:5001/stream")
        src = f'souphttpsrc location="{url}" is-live=true ! multipartdemux ! jpegdec'
    else:
        device = cam_cfg.get("device", "/dev/video0")
        src = f'v4l2src device={device}'

    pipe_str = (
        f'{src} ! imxvideoconvert_g2d '
        f'! video/x-raw,format=RGB '
        f'! appsink name=sink sync=false max-buffers=2 drop=true'
    )
    pipeline = None
    try:
        pipeline = Gst.parse_launch(pipe_str)
        appsink  = pipeline.get_by_name('sink')
        pipeline.set_state(Gst.State.PLAYING)
        ret, _, _ = pipeline.get_state(2 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline state change FAILURE")
        logger.info("GStreamer G2D pipeline started.")
        return pipeline, appsink
    except Exception as e:
        logger.warning(f"GStreamer failed ({e}), falling back to OpenCV.")
        if pipeline:
            pipeline.set_state(Gst.State.NULL)
        return None, None


def _gst_capture_loop(appsink, preproc_queue: Queue,
                      input_h: int, input_w: int,
                      input_dtype, stop_event: threading.Event) -> None:
    import gi
    from gi.repository import Gst

    while not stop_event.is_set():
        t0 = time.monotonic()
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

        bgr = rgb[:, :, ::-1].copy()

        if h != input_h or w != input_w:
            rgb_small = cv2.resize(rgb, (input_w, input_h))
        else:
            rgb_small = rgb

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
    logger.debug("GStreamer capture thread stopped.")


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
        item = (frame, out_data, invoke_ms_acc + npu_ms)
        if out_queue.full():
            try:
                out_queue.get_nowait()
            except Empty:
                pass
        out_queue.put_nowait(item)
    logger.debug(f"Stage '{label}' thread stopped.")


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(config: dict) -> None:
    cam_cfg    = config["camera"]
    stop_event = threading.Event()

    if cam_cfg.get('source', 'local') == 'local' and not cam_cfg.get('device'):
        candidates = _list_cameras()
        if not candidates:
            logger.error("No USB cameras found.")
            sys.exit(1)
        random.shuffle(candidates)
        chosen = None
        for cam in candidates:
            probe = cv2.VideoCapture(cam['device'])
            ok = probe.isOpened() and probe.grab()
            probe.release()
            if ok:
                chosen = cam
                break
        if chosen is None:
            logger.error("No working USB camera found.")
            sys.exit(1)
        cam_cfg['device'] = chosen['device']
        logger.info(f"Auto-selected: {chosen['name']} ({chosen['device']})")
        set_auto_exposure_on_usb_cameras(cam_cfg['device'])

    # --- Pipeline ---
    logger.info("Loading pipeline...")
    pipeline = inference.load_pipeline(config)
    for stage in pipeline:
        stage["avg_ms"] = 0.0
    labels     = inference.load_labels(config["model"]["labels_path"])
    model_name = Path(config["model"]["path"]).stem
    logger.info(f"Labels: {len(labels)} | Stages: {len(pipeline)}")

    inp_det          = pipeline[0]["input_details"][0]
    input_h, input_w = inp_det['shape'][1], inp_det['shape'][2]
    input_dtype      = inp_det['dtype']

    # --- Performance logger ---
    perf = perf_logger.PerfLogger(
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
        logger.info("BYTETracker devre dışı (config: tracking.enabled=false)")

    # --- Camera ---
    preproc_queue: Queue = Queue(maxsize=2)
    gst_pipe, appsink   = _open_gst_pipeline(cam_cfg)
    capture_state: dict = {'stop': threading.Event(), 'gst_pipe': gst_pipe}

    def _start_opencv_capture(device: str) -> None:
        cam_cfg['device'] = device
        cap, desc = _open_capture(cam_cfg)
        logger.info(f"Camera: {desc}")
        fq: Queue = Queue(maxsize=2)
        s = capture_state['stop']
        threading.Thread(target=_capture_loop, args=(cap, fq, s),
                         daemon=True, name='FrameCapture').start()
        threading.Thread(target=_preprocess_loop,
                         args=(fq, preproc_queue, pipeline[0]['input_details'], s),
                         daemon=True, name='Preprocess').start()

    def _switch_camera(new_device: str) -> None:
        logger.info(f"Camera switch: {new_device}")
        probe = cv2.VideoCapture(new_device)
        if not probe.isOpened():
            logger.error(f"Cannot open {new_device}")
            probe.release()
            return
        probe.release()
        gp = capture_state.get('gst_pipe')
        if gp is not None:
            try:
                import gi
                from gi.repository import Gst
                gp.set_state(Gst.State.NULL)
            except Exception:
                pass
            capture_state['gst_pipe'] = None
        capture_state['stop'].set()
        time.sleep(0.2)
        capture_state['stop'] = threading.Event()
        _start_opencv_capture(new_device)
        streamer.set_active_camera(new_device)

    if gst_pipe is not None:
        threading.Thread(
            target=_gst_capture_loop,
            args=(appsink, preproc_queue, input_h, input_w, input_dtype, capture_state['stop']),
            daemon=True, name="GstCapture"
        ).start()
        logger.info("GStreamer G2D capture started.")
    else:
        _start_opencv_capture(cam_cfg.get('device', '/dev/video0'))

    streamer.set_active_camera(cam_cfg.get('device', ''))
    streamer.register_camera_callbacks(_list_cameras, _switch_camera)

    stage_in_queue = preproc_queue
    for stage in pipeline:
        stage_out_queue: Queue = Queue(maxsize=2)
        threading.Thread(
            target=_stage_loop,
            args=(stage, stage_in_queue, stage_out_queue, stop_event),
            daemon=True, name=f"Stage-{stage['label']}"
        ).start()
        logger.info(f"Stage '{stage['label']}' started.")
        stage_in_queue = stage_out_queue
    raw_queue = stage_in_queue

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

    # EMA display stats
    fps_ema          = 0.0
    invoke_ms_avg    = 0.0
    inference_ms_avg = 0.0
    frame_count      = 0
    t_start          = time.monotonic()

    # -----------------------------------------------------------------
    # Doğru FPS ölçümü için: iki ardışık frame alımı arasındaki süreyi
    # ölçeriz. t_prev_frame = önceki frame'in kuyruktan çıktığı an.
    # -----------------------------------------------------------------
    t_prev_frame: float | None = None

    logger.info("Inference loop started. Ctrl+C to stop.")

    try:
        while True:
            # --- Frame al (blocking) ---
            try:
                frame, raw_list, invoke_ms = raw_queue.get(timeout=1.0)
            except Empty:
                logger.warning("No frame in 1s — waiting for camera...")
                continue

            t_got = time.monotonic()   # frame'in kuyruktan çıktığı an

            # Gerçek frame-to-frame süresi
            if t_prev_frame is not None:
                frame_ms = (t_got - t_prev_frame) * 1000.0
            else:
                frame_ms = 33.3  # ilk frame için tahmin (30fps)
            t_prev_frame = t_got

            # Decode süresi (capture thread'den)
            decode_ms = _decode_ms_shared[0]

            # Dequantize — TFLite int8 only.
            # RKNN invoke_stage() returns a Python list of 9 float32 tensors;
            # isinstance guard prevents AttributeError on .dtype for that case.
            raw = raw_list[0]
            if isinstance(raw, np.ndarray) and raw.dtype == np.int8:
                last_out_det = pipeline[-1]["output_details"]
                scale, zero_point = last_out_det[0]["quantization"]
                raw = (raw.astype(np.float32) - zero_point) * scale

            # Postprocess — bu süreyı ayrıca ölçüyoruz
            t_post = time.monotonic()
            if config["inference"].get("skip_postprocess", False):
                detections = []
            else:
                detections = inference.postprocess_detections(raw, frame, config)
            postproc_ms = (time.monotonic() - t_post) * 1000

            # --- BYTETracker güncelle ---
            if tracker is not None:
                tracks = tracker.update(detections)
            else:
                tracks = None

            # --- Perf güncelle ---
            perf.update(
                frame_ms   = frame_ms,
                decode_ms  = decode_ms,
                invoke_ms  = invoke_ms,
                postproc_ms= postproc_ms,
            )

            # Annotate + stream
            annotated = overlay.annotate_frame(
                frame, detections, labels, fps_ema, config,
                model_name, postproc_ms, invoke_ms,
                tracks=tracks      # None ise ham dedektör kutuları gösterilir
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

            # EMA display
            instant_fps   = 1000.0 / frame_ms if frame_ms > 0 else 0.0
            fps_ema       += _STATS_ALPHA * (instant_fps  - fps_ema)
            invoke_ms_avg += _STATS_ALPHA * (invoke_ms    - invoke_ms_avg)
            inference_ms_avg += _STATS_ALPHA * (postproc_ms - inference_ms_avg)

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
                    f"NPU≈{st['npu_pct']['avg']:.0f}% | "
                    f"RAM={st['ram_mb']['avg']:.0f}MB | "
                    f"up={uptime:.0f}s"
                )

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        stop_event.set()
        capture_state['stop'].set()
        gp = capture_state.get('gst_pipe')
        if gp is not None:
            try:
                import gi
                from gi.repository import Gst
                gp.set_state(Gst.State.NULL)
            except Exception:
                pass
        perf.flush()
        logger.info("\n" + perf.summary())
        logger.info(f"Metrics → {_PERF_LOG_PATH}")
        if tracker is not None:
            logger.info(tracker.summary())
        logger.info("Stopped. Goodbye.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FRDM-IMX95 Edge AI Vision"
    )
    parser.add_argument("--config",  default="config.json")
    parser.add_argument("--no-npu",  action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    resolve_model_path(config)
    logger.info(f"Model path: {config['model']['path']}")

    if args.no_npu:
        config["model"]["use_npu"] = False
        logger.info("NPU disabled via --no-npu flag")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run(config)


if __name__ == "__main__":
    main()
