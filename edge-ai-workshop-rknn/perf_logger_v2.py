"""
perf_logger_v2.py — Performance metrics tracker for RK3588 Edge AI Workshop.
Includes Hardware Counters for NPU (debugfs) and Power (hwmon).

Metrics (standart):
    fps             — End-to-end FPS
    frame_ms        — Gerçek frame-to-frame süresi (ms)
    decode_ms       — GStreamer / OpenCV capture + decode süresi (ms)
    invoke_ms       — NPU inference invoke süresi (ms)
    postproc_ms     — CPU postprocess süresi (ms)
    cpu_system_pct  — Toplam sistem CPU kullanımı (%)
    process_cpu_pct — Bu process'in normalize CPU kullanımı (%)
    background_cpu_pct — Sistem - process CPU (arka plan yükü) (%)
    cpu_pct         — cpu_system_pct alias (main_v2.py uyumluluğu için)
    npu_pct         — NPU kullanım tahmini: invoke_ms / frame_ms * 100
    ram_mb          — Uygulama RAM kullanımı (RSS, MB)
    temp_c          — CPU sıcaklığı (°C)
    cpu_core_N_pct  — N. çekirdeğin kullanımı (%, dinamik)

Metrics (RK3588 donanım):
    npu_core0/1/2   — Gerçek NPU donanım kullanımı (%, debugfs peak sampling)
    power_w         — Sistem anlık güç tüketimi (W, hwmon)
"""

from __future__ import annotations

import csv
import glob
import os
import re
import threading
import time
from collections import deque
from datetime import datetime

import psutil

METRICS = [
    'fps', 'frame_ms', 'decode_ms', 'invoke_ms', 'postproc_ms',
    'cpu_system_pct', 'process_cpu_pct', 'background_cpu_pct', 'cpu_pct',
    'npu_pct', 'ram_mb', 'temp_c',
    'npu_core0', 'npu_core1', 'npu_core2', 'power_w',
]

UNITS = {
    'fps':                'fps',
    'frame_ms':           'ms',
    'decode_ms':          'ms',
    'invoke_ms':          'ms',
    'postproc_ms':        'ms',
    'cpu_system_pct':     '%',
    'process_cpu_pct':    '%',
    'background_cpu_pct': '%',
    'cpu_pct':            '%',
    'npu_pct':            '%',
    'ram_mb':             'MB',
    'temp_c':             '°C',
    'npu_core0':          '%',
    'npu_core1':          '%',
    'npu_core2':          '%',
    'power_w':            'W',
}


# ---------------------------------------------------------------------------
# RK3588 NPU donanım sayacı
# ---------------------------------------------------------------------------

class NPUMonitor:
    """
    Arka planda 5ms'de bir /sys/kernel/debug/rknpu/load okuyarak
    peak NPU yükünü yakalar. RK3588'e özgüdür; dosya yoksa N/A döner.
    """

    def __init__(self, interval_ms: int = 5):
        self._interval  = interval_ms / 1000.0
        self._stop      = threading.Event()
        self._peaks     = [0, 0, 0]
        self._lock      = threading.Lock()
        self._ever_read = False
        self._warned    = False
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="NPUMonitor")
        self._thread.start()

    def _run(self) -> None:
        core_re = re.compile(r"Core(\d+):\s*(\d+)%")
        while not self._stop.is_set():
            try:
                with open("/sys/kernel/debug/rknpu/load") as f:
                    text = f.read()
                matches = core_re.findall(text)
                if matches:
                    with self._lock:
                        for core_str, val_str in matches:
                            idx, val = int(core_str), int(val_str)
                            if 0 <= idx < 3:
                                self._peaks[idx] = max(self._peaks[idx], val)
                        self._ever_read = True
            except PermissionError:
                if not self._warned:
                    print("UYARI: NPU yükü okunamıyor — sudo ile çalıştır. npu_core N/A.")
                    self._warned = True
            except FileNotFoundError:
                if not self._warned:
                    print("UYARI: /sys/kernel/debug/rknpu/load bulunamadı. npu_core N/A.")
                    self._warned = True
            time.sleep(self._interval)

    def get_peak_and_reset(self) -> list[float]:
        with self._lock:
            if not self._ever_read:
                return [-1.0, -1.0, -1.0]
            peaks = [float(p) for p in self._peaks]
            self._peaks = [0, 0, 0]
            return peaks

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Güç tüketimi (hwmon)
# ---------------------------------------------------------------------------

class PowerMonitor:
    """
    sysfs hwmon üzerinden INA226/INA231 veya TCPM güç ölçümlerini okur.
    Desteklenen donanım yoksa -1.0 döner.
    """

    def __init__(self):
        self._direct_path  = ""
        self._voltage_path = ""
        self._current_path = ""
        self.chip_name     = ""
        self._detect()

    def _detect(self) -> None:
        for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
            try:
                with open(os.path.join(hwmon, "name")) as f:
                    chip = f.read().strip()
            except IOError:
                continue
            for i in range(4):
                cand = os.path.join(hwmon, f"power{i}_input")
                if os.path.exists(cand):
                    self._direct_path = cand
                    self.chip_name    = chip
                    return
            vp = os.path.join(hwmon, "in0_input")
            cp = os.path.join(hwmon, "curr1_input")
            if os.path.exists(vp) and os.path.exists(cp) and not self._voltage_path:
                self._voltage_path = vp
                self._current_path = cp
                self.chip_name     = chip

    def read_watts(self) -> float:
        try:
            if self._direct_path:
                with open(self._direct_path) as f:
                    return float(f.read().strip()) / 1e6
            if self._voltage_path and self._current_path:
                with open(self._voltage_path) as fv, open(self._current_path) as fc:
                    return (float(fv.read().strip()) / 1000.0) * \
                           (float(fc.read().strip()) / 1000.0)
        except IOError:
            pass
        return -1.0


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

class PerfWindow:
    """Thread-safe sliding window — min / max / avg. -1.0 (N/A) değerleri atlar."""

    def __init__(self, maxsize: int = 300):
        self._data: deque[float] = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def update(self, value: float) -> None:
        if value < 0:
            return
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


# ---------------------------------------------------------------------------
# Ana logger
# ---------------------------------------------------------------------------

class PerfLoggerV2:
    """
    RK3588 için genişletilmiş perf logger.
    main_v2.py arayüzüyle tam uyumludur (PerfLoggerV2, METRICS, stop()).

    Kullanım:
        perf = PerfLoggerV2(...)
        perf.update(frame_ms=..., decode_ms=..., invoke_ms=..., postproc_ms=...)
        perf.flush()
        print(perf.summary())
        perf.stop()
    """

    def __init__(self,
                 log_path:    str = 'perf_metrics.csv',
                 window:      int = 300,
                 flush_every: int = 100):
        self._windows      = {m: PerfWindow(window) for m in METRICS}
        self._log_path     = log_path
        self._flush_every  = flush_every
        self._frame_count  = 0
        self._lock         = threading.Lock()
        self._process      = psutil.Process(os.getpid())
        self._core_count   = psutil.cpu_count(logical=True)
        self._thermal_path = self._find_thermal_zone()
        self._start_time: float | None = None   # ilk update'te set edilir

        # Per-core windows — dinamik çekirdek sayısı
        self._core_windows: list[PerfWindow] = [
            PerfWindow(window) for _ in range(self._core_count)
        ]
        for i in range(self._core_count):
            UNITS[self._core_metric_name(i)] = '%'

        # ARM HZ=100 jiffy artifact önlemi: 500ms'de bir örnekle
        self._cpu_sample_interval: float     = 0.5
        self._last_cpu_sample_t:   float     = 0.0
        self._last_core_pcts:      list[float] = [0.0] * self._core_count
        self._last_process_cpu:    float     = 0.0

        # RK3588 donanım sayaçları
        self._npu_monitor   = NPUMonitor(interval_ms=5)
        self._power_monitor = PowerMonitor()

        # Warmup
        psutil.cpu_percent(interval=None, percpu=True)
        self._process.cpu_percent(interval=None)

        self._init_csv()

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    def _find_thermal_zone(self, target_type: str = "soc-thermal") -> str:
        """RK3588 termal zone — soc-thermal."""
        base = "/sys/class/thermal/"
        if not os.path.exists(base):
            return ""
        try:
            for zone in os.listdir(base):
                if not zone.startswith("thermal_zone"):
                    continue
                tp = os.path.join(base, zone, "type")
                if os.path.exists(tp):
                    with open(tp) as f:
                        if f.read().strip() == target_type:
                            return os.path.join(base, zone, "temp")
        except Exception as e:
            print(f"Uyarı: Termal sensör taranırken hata: {e}")
        return ""

    def _read_temperature(self) -> float:
        if not self._thermal_path:
            return 0.0
        try:
            with open(self._thermal_path) as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return 0.0

    def _core_metric_name(self, idx: int) -> str:
        return f"cpu_core_{idx}_pct"

    # ------------------------------------------------------------------
    # CSV başlığı
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        with open(self._log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'frame_count']
            for m in METRICS:
                header += [f'{m}_avg', f'{m}_min', f'{m}_max']
            for i in range(self._core_count):
                m = self._core_metric_name(i)
                header += [f'{m}_avg', f'{m}_min', f'{m}_max']
            writer.writerow(header)

    # ------------------------------------------------------------------
    # Güncelleme
    # ------------------------------------------------------------------

    def update(self, frame_ms: float, decode_ms: float,
               invoke_ms: float, postproc_ms: float) -> None:
        if self._start_time is None:
            self._start_time = time.monotonic()

        fps     = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        ram_mb  = self._process.memory_info().rss / 1024 / 1024
        npu_pct = (invoke_ms / frame_ms * 100.0) if frame_ms > 0 else 0.0
        temp_c  = self._read_temperature()

        # CPU — 500ms'de bir örnekle (jiffy artifact önlemi)
        now = time.monotonic()
        if now - self._last_cpu_sample_t >= self._cpu_sample_interval:
            self._last_core_pcts   = psutil.cpu_percent(interval=None, percpu=True)
            self._last_process_cpu = (
                self._process.cpu_percent(interval=None) / self._core_count
            )
            self._last_cpu_sample_t = now
        per_core_pcts      = self._last_core_pcts
        cpu_system_pct     = sum(per_core_pcts) / len(per_core_pcts)
        process_cpu_pct    = self._last_process_cpu
        background_cpu_pct = max(0.0, cpu_system_pct - process_cpu_pct)

        # RK3588 donanım sayaçları
        npu_peaks = self._npu_monitor.get_peak_and_reset()
        power_w   = self._power_monitor.read_watts()

        values = {
            'fps':                fps,
            'frame_ms':           frame_ms,
            'decode_ms':          decode_ms,
            'invoke_ms':          invoke_ms,
            'postproc_ms':        postproc_ms,
            'cpu_system_pct':     cpu_system_pct,
            'process_cpu_pct':    process_cpu_pct,
            'background_cpu_pct': background_cpu_pct,
            'cpu_pct':            cpu_system_pct,   # main_v2.py alias
            'npu_pct':            npu_pct,
            'ram_mb':             ram_mb,
            'temp_c':             temp_c,
            'npu_core0':          npu_peaks[0],
            'npu_core1':          npu_peaks[1],
            'npu_core2':          npu_peaks[2],
            'power_w':            power_w,
        }
        for m, v in values.items():
            self._windows[m].update(v)

        for i, pct in enumerate(per_core_pcts):
            self._core_windows[i].update(pct)

        with self._lock:
            self._frame_count += 1
            should_flush = (self._frame_count % self._flush_every == 0)

        if should_flush:
            self.flush()

    # ------------------------------------------------------------------
    # Flush & özet
    # ------------------------------------------------------------------

    def flush(self) -> dict:
        stats = {m: self._windows[m].stats() for m in METRICS}
        core_stats = {
            self._core_metric_name(i): self._core_windows[i].stats()
            for i in range(self._core_count)
        }

        with open(self._log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [datetime.now().isoformat(), self._frame_count]
            for m in METRICS:
                s = stats[m]
                row += [f"{s['avg']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}"]
            for i in range(self._core_count):
                s = core_stats[self._core_metric_name(i)]
                row += [f"{s['avg']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}"]
            writer.writerow(row)

        return {**stats, **core_stats}

    def summary(self) -> str:
        hw_metrics   = ['npu_core0', 'npu_core1', 'npu_core2', 'power_w']
        # cpu_pct alias'ı summary'de gösterme (cpu_system_pct ile aynı)
        skip         = {'cpu_pct'}
        base_metrics = [m for m in METRICS if m not in hw_metrics and m not in skip]

        def fmt_row(name: str, s: dict, unit: str) -> str:
            if s['count'] == 0:
                return f"  {name:<18} {'N/A':>10}     {'N/A':>10}     {'N/A':>10}"
            return (
                f"  {name:<18} "
                f"{s['avg']:>8.1f}{unit:4s} "
                f"{s['min']:>8.1f}{unit:4s} "
                f"{s['max']:>8.1f}{unit:4s}"
            )

        lines = [
            '=' * 64,
            '  PERFORMANCE SUMMARY',
            '=' * 64,
            f"  {'Metric':<18} {'AVG':>10} {'MIN':>10} {'MAX':>10}",
            '  ' + '-' * 60,
        ]

        for m in base_metrics:
            lines.append(fmt_row(m, self._windows[m].stats(), UNITS[m]))

        # Per-core CPU
        lines.append('  ' + '-' * 60)
        lines.append('  Per-Core CPU Usage')
        lines.append('  ' + '-' * 60)
        for i in range(self._core_count):
            name = self._core_metric_name(i)
            lines.append(fmt_row(name, self._core_windows[i].stats(), UNITS[name]))

        # RK3588 donanım sayaçları
        lines.append('  ' + '-' * 60)
        lines.append('  Hardware Counters (RK3588)')
        lines.append('  ' + '-' * 60)
        for m in hw_metrics:
            lines.append(fmt_row(m, self._windows[m].stats(), UNITS[m]))

        duration_s   = time.monotonic() - (self._start_time or time.monotonic())
        duration_str = f"{int(duration_s // 60)}m {int(duration_s % 60)}s"

        lines += [
            '  ' + '-' * 60,
            f"  Window size  : {self._windows['fps'].stats()['count']} frames",
            f"  Total frames : {self._frame_count}",
            f"  Duration     : {duration_str}",
            f"  Logical cores: {self._core_count}",
            '=' * 64,
        ]
        return '\n'.join(lines)

    def stop(self) -> None:
        """Kapanırken NPUMonitor arka plan thread'ini durdur."""
        self._npu_monitor.stop()
