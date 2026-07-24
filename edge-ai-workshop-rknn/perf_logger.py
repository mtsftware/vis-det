"""
perf_logger.py — Performance metrics tracker for Edge AI Workshop.

Metrics:
    fps         — End-to-end FPS (frame interval ölçümü, doğru)
    frame_ms    — Gerçek frame-to-frame süresi (ms)
    decode_ms   — GStreamer / OpenCV capture + decode süresi (ms)
    invoke_ms   — NPU inference invoke süresi (ms)
    postproc_ms — CPU postprocess süresi (ms)
    cpu_pct     — Sistem CPU kullanımı (%)
    npu_pct     — NPU kullanım tahmini: invoke_ms / frame_ms * 100
    ram_mb      — Uygulama RAM kullanımı (RSS, MB)
"""


from __future__ import annotations

import csv
import os
import threading
from collections import deque
from datetime import datetime

import psutil

METRICS = ['fps', 'frame_ms', 'decode_ms', 'invoke_ms',
           'postproc_ms', 'cpu_pct', 'npu_pct', 'ram_mb']

UNITS = {
    'fps':        'fps',
    'frame_ms':   'ms',
    'decode_ms':  'ms',
    'invoke_ms':  'ms',
    'postproc_ms':'ms',
    'cpu_pct':    '%',
    'npu_pct':    '%',
    'ram_mb':     'MB',
}


class PerfWindow:
    """Thread-safe sliding window — min / max / avg."""

    def __init__(self, maxsize: int = 300):
        self._data: deque[float] = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def update(self, value: float) -> None:
        with self._lock:
            self._data.append(value)

    def stats(self) -> dict:
        with self._lock:
            if not self._data:
                return {'min': 0.0, 'max': 0.0, 'avg': 0.0, 'count': 0}
            data = list(self._data)
        return {
            'min':   min(data),
            'max':   max(data),
            'avg':   sum(data) / len(data),
            'count': len(data),
        }


class PerfLogger:
    """
    Collects per-frame measurements and writes min/max/avg to CSV
    every `flush_every` frames.

    Doğru kullanım:
        perf = PerfLogger(...)

        # Her frame'de:
        perf.update(
            frame_ms   = gerçek frame-to-frame süresi,
            decode_ms  = capture thread decode süresi,
            invoke_ms  = NPU invoke süresi,
            postproc_ms= CPU postprocess süresi,
        )

        # Sonunda:
        perf.flush()
        print(perf.summary())
    """

    def __init__(self,
                 log_path: str = 'perf_metrics.csv',
                 window: int = 300,
                 flush_every: int = 100):
        self._windows     = {m: PerfWindow(window) for m in METRICS}
        self._log_path    = log_path
        self._flush_every = flush_every
        self._frame_count = 0
        self._lock        = threading.Lock()
        self._process     = psutil.Process(os.getpid())

        # İlk çağrı her zaman 0.0 döner, önden çağır
        psutil.cpu_percent(interval=None)

        self._init_csv()

    def _init_csv(self) -> None:
        with open(self._log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'frame_count']
            for m in METRICS:
                header += [f'{m}_avg', f'{m}_min', f'{m}_max']
            writer.writerow(header)

    def update(self, frame_ms: float, decode_ms: float,
               invoke_ms: float, postproc_ms: float) -> None:
        """
        frame_ms   : iki ardışık frame alımı arasındaki süre (gerçek FPS kaynağı)
        decode_ms  : capture thread'den gelen decode süresi
        invoke_ms  : NPU invoke süresi (stage thread'den)
        postproc_ms: CPU postprocess süresi (main loop'ta ölçülür)
        """
        fps     = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_mb  = self._process.memory_info().rss / 1024 / 1024
        # NPU % = NPU'nun bir frame süresinin ne kadarını kullandığı
        npu_pct = (invoke_ms / frame_ms * 100.0) if frame_ms > 0 else 0.0

        values = {
            'fps':        fps,
            'frame_ms':   frame_ms,
            'decode_ms':  decode_ms,
            'invoke_ms':  invoke_ms,
            'postproc_ms':postproc_ms,
            'cpu_pct':    cpu_pct,
            'npu_pct':    npu_pct,
            'ram_mb':     ram_mb,
        }
        for m, v in values.items():
            self._windows[m].update(v)

        with self._lock:
            self._frame_count += 1
            should_flush = (self._frame_count % self._flush_every == 0)

        if should_flush:
            self.flush()

    def flush(self) -> dict:
        stats = {m: self._windows[m].stats() for m in METRICS}
        with open(self._log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [datetime.now().isoformat(), self._frame_count]
            for m in METRICS:
                s = stats[m]
                row += [f"{s['avg']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}"]
            writer.writerow(row)
        return stats

    def summary(self) -> str:
        lines = [
            '=' * 64,
            '  PERFORMANCE SUMMARY',
            '=' * 64,
            f"  {'Metric':<14} {'AVG':>10} {'MIN':>10} {'MAX':>10}",
            '  ' + '-' * 60,
        ]
        for m in METRICS:
            s    = self._windows[m].stats()
            unit = UNITS[m]
            lines.append(
                f"  {m:<14} "
                f"{s['avg']:>8.1f}{unit:4s} "
                f"{s['min']:>8.1f}{unit:4s} "
                f"{s['max']:>8.1f}{unit:4s}"
            )
        lines += [
            '  ' + '-' * 60,
            f"  Window size  : {self._windows['fps'].stats()['count']} frames",
            f"  Total frames : {self._frame_count}",
            '=' * 64,
        ]
        return '\n'.join(lines)
