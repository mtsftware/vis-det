#include "perf/PerfLogger.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// Metrik tablosu — perf_logger_v2.py'deki METRICS/UNITS listeleriyle
// BİREBİR AYNI sıra (CSV kolon uyumluluğu için sıra önemli).
// ---------------------------------------------------------------------
enum Metric {
    kFps = 0,
    kFrameMs,
    kDecodeMs,
    kInvokeMs,
    kPostprocMs,
    kCpuSystemPct,
    kProcessCpuPct,
    kBackgroundCpuPct,
    kCpuPct,
    kNpuPct,
    kRamMb,
    kTempC,
    kNpuCore0,
    kNpuCore1,
    kNpuCore2,
    kPowerW,
    kMetricCount,
};

const char* kMetricNames[kMetricCount] = {
    "fps", "frame_ms", "decode_ms", "invoke_ms", "postproc_ms",
    "cpu_system_pct", "process_cpu_pct", "background_cpu_pct", "cpu_pct",
    "npu_pct", "ram_mb", "temp_c",
    "npu_core0", "npu_core1", "npu_core2", "power_w",
};

const char* kMetricUnits[kMetricCount] = {
    "fps", "ms", "ms", "ms", "ms",
    "%", "%", "%", "%",
    "%", "MB", "C",
    "%", "%", "%", "W",
};

std::string readFileTrim(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::string line;
    std::getline(f, line);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    return line;
}

bool readFileDouble(const std::string& path, double& out) {
    std::string s = readFileTrim(path);
    if (s.empty()) return false;
    try {
        out = std::stod(s);
        return true;
    } catch (...) {
        return false;
    }
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return static_cast<bool>(f);
}

uint64_t nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

std::string isoTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  now.time_since_epoch()) % std::chrono::seconds(1);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0')
        << std::setw(6) << us.count();
    return oss.str();
}

// ---------------------------------------------------------------------
// Sliding window — min/max/avg. Negatif (N/A) değerler atlanır (NPUMonitor/
// PowerMonitor donanım yoksa -1.0 döndürüyor, perf_logger_v2.py ile aynı
// sözleşme).
// ---------------------------------------------------------------------
class PerfWindow {
public:
    explicit PerfWindow(size_t maxsize = 300) : maxsize_(maxsize) {}

    // PerfWindow bir std::mutex barındırdığı için kopyalanamaz/taşınamaz —
    // pencere boyutu kurulumdan SONRA bu metodla ayarlanır (Impl ctor'unda
    // atama/vector::assign yerine kullanılıyor).
    void setMaxSize(size_t maxsize) {
        std::lock_guard<std::mutex> lk(mutex_);
        maxsize_ = maxsize;
        while (data_.size() > maxsize_) data_.pop_front();
    }

    void update(double value) {
        if (value < 0) return;
        std::lock_guard<std::mutex> lk(mutex_);
        data_.push_back(value);
        while (data_.size() > maxsize_) data_.pop_front();
    }

    struct Stats {
        double min = 0.0, max = 0.0, avg = 0.0;
        size_t count = 0;
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lk(mutex_);
        Stats s;
        s.count = data_.size();
        if (data_.empty()) return s;
        s.min = *std::min_element(data_.begin(), data_.end());
        s.max = *std::max_element(data_.begin(), data_.end());
        double sum = 0.0;
        for (double v : data_) sum += v;
        s.avg = sum / data_.size();
        return s;
    }

private:
    size_t maxsize_;
    mutable std::mutex mutex_;
    std::deque<double> data_;
};

// ---------------------------------------------------------------------
// RK3588 NPU donanım sayacı — /sys/kernel/debug/rknpu/load'u 5ms'de bir
// okuyup peak yükü yakalar (perf_logger_v2.py::NPUMonitor ile aynı).
// ---------------------------------------------------------------------
class NPUMonitor {
public:
    explicit NPUMonitor(int interval_ms = 5) : interval_ms_(interval_ms) {
        thread_ = std::thread([this] { run(); });
    }

    ~NPUMonitor() { stop(); }

    void stop() {
        if (stop_.exchange(true)) return;
        if (thread_.joinable()) thread_.join();
    }

    // -1.0,-1.0,-1.0: dosya hiç okunamadı (donanım yok / izin yok).
    void getPeakAndReset(double out[3]) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!ever_read_) {
            out[0] = out[1] = out[2] = -1.0;
            return;
        }
        for (int i = 0; i < 3; ++i) out[i] = peaks_[i];
        peaks_[0] = peaks_[1] = peaks_[2] = 0.0;
    }

private:
    void run() {
        while (!stop_.load()) {
            std::ifstream f("/sys/kernel/debug/rknpu/load");
            if (f) {
                std::stringstream ss;
                ss << f.rdbuf();
                sample(ss.str());
            } else if (!warned_) {
                std::cerr << "[PerfLogger] UYARI: /sys/kernel/debug/rknpu/load "
                             "okunamıyor (izin ya da donanım yok) — npu_core N/A.\n";
                warned_ = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        }
    }

    void sample(const std::string& text) {
        size_t pos = 0;
        bool any = false;
        while ((pos = text.find("Core", pos)) != std::string::npos) {
            size_t idx_pos = pos + 4;
            if (idx_pos >= text.size() || !isdigit(static_cast<unsigned char>(text[idx_pos]))) {
                pos += 4;
                continue;
            }
            int core_idx = text[idx_pos] - '0';
            size_t colon = text.find(':', idx_pos);
            if (colon == std::string::npos) break;
            size_t p = colon + 1;
            while (p < text.size() && (text[p] == ' ' || text[p] == '\t')) ++p;
            size_t num_start = p;
            while (p < text.size() && isdigit(static_cast<unsigned char>(text[p]))) ++p;
            if (p > num_start && core_idx >= 0 && core_idx < 3) {
                int val = std::stoi(text.substr(num_start, p - num_start));
                std::lock_guard<std::mutex> lk(mutex_);
                peaks_[core_idx] = std::max(peaks_[core_idx], static_cast<double>(val));
                any = true;
            }
            pos = idx_pos + 1;
        }
        if (any) {
            std::lock_guard<std::mutex> lk(mutex_);
            ever_read_ = true;
        }
    }

    int interval_ms_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    std::mutex mutex_;
    double peaks_[3] = {0.0, 0.0, 0.0};
    bool ever_read_ = false;
    bool warned_ = false;
};

// ---------------------------------------------------------------------
// Güç tüketimi (hwmon) — perf_logger_v2.py::PowerMonitor ile aynı algoritma:
// önce doğrudan powerN_input (µW), yoksa in0_input(mV) * curr1_input(mA).
// ---------------------------------------------------------------------
class PowerMonitor {
public:
    PowerMonitor() { detect(); }

    double readWatts() const {
        if (!direct_path_.empty()) {
            double v;
            if (readFileDouble(direct_path_, v)) return v / 1e6;
        }
        if (!voltage_path_.empty() && !current_path_.empty()) {
            double v, c;
            if (readFileDouble(voltage_path_, v) && readFileDouble(current_path_, c)) {
                return (v / 1000.0) * (c / 1000.0);
            }
        }
        return -1.0;
    }

private:
    void detect() {
        DIR* dir = opendir("/sys/class/hwmon");
        if (!dir) return;
        std::vector<std::string> hwmon_dirs;
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (name.rfind("hwmon", 0) == 0) {
                hwmon_dirs.push_back("/sys/class/hwmon/" + name);
            }
        }
        closedir(dir);
        std::sort(hwmon_dirs.begin(), hwmon_dirs.end());

        for (const auto& hwmon : hwmon_dirs) {
            std::string chip = readFileTrim(hwmon + "/name");
            if (chip.empty()) continue;

            bool found_direct = false;
            for (int i = 0; i < 4; ++i) {
                std::string cand = hwmon + "/power" + std::to_string(i) + "_input";
                if (fileExists(cand)) {
                    direct_path_ = cand;
                    found_direct = true;
                    break;
                }
            }
            if (found_direct) return;

            std::string vp = hwmon + "/in0_input";
            std::string cp = hwmon + "/curr1_input";
            if (fileExists(vp) && fileExists(cp) && voltage_path_.empty()) {
                voltage_path_ = vp;
                current_path_ = cp;
            }
        }
    }

    std::string direct_path_;
    std::string voltage_path_;
    std::string current_path_;
};

// ---------------------------------------------------------------------
// CPU/RAM/sıcaklık okuyucuları — psutil'in C++'ta karşılığı yok, /proc ve
// /sys doğrudan okunuyor.
// ---------------------------------------------------------------------
struct CpuTimes {
    uint64_t idle_all = 0;  // idle + iowait
    uint64_t total = 0;
};

bool readProcStatLine(std::istream& in, std::string& label, CpuTimes& out) {
    std::string line;
    if (!std::getline(in, line)) return false;
    std::istringstream ss(line);
    ss >> label;
    uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0,
             steal = 0;
    ss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    out.idle_all = idle + iowait;
    uint64_t non_idle = user + nice + system + irq + softirq + steal;
    out.total = out.idle_all + non_idle;
    return true;
}

// /proc/self/stat: pid (comm) state ppid ... utime(14) stime(15) ...
// comm parantez içinde boşluk barındırabildiği için son ')' baz alınır.
bool readProcessJiffies(uint64_t& utime_stime) {
    std::ifstream f("/proc/self/stat");
    if (!f) return false;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t close_paren = content.rfind(')');
    if (close_paren == std::string::npos) return false;
    std::istringstream ss(content.substr(close_paren + 1));
    std::vector<std::string> fields;
    std::string tok;
    while (ss >> tok) fields.push_back(tok);
    // fields[0] = state (field 3), utime field14 -> index 11, stime field15 -> index 12
    if (fields.size() < 13) return false;
    try {
        uint64_t utime = std::stoull(fields[11]);
        uint64_t stime = std::stoull(fields[12]);
        utime_stime = utime + stime;
        return true;
    } catch (...) {
        return false;
    }
}

double readRssMb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream ss(line.substr(6));
            double kb = 0;
            ss >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
}

std::string findThermalZone(const std::string& target_type = "soc-thermal") {
    const std::string base = "/sys/class/thermal/";
    DIR* dir = opendir(base.c_str());
    if (!dir) return "";
    struct dirent* entry;
    std::string result;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.rfind("thermal_zone", 0) != 0) continue;
        std::string type = readFileTrim(base + name + "/type");
        if (type == target_type) {
            result = base + name + "/temp";
            break;
        }
    }
    closedir(dir);
    return result;
}

double readTemperature(const std::string& path) {
    if (path.empty()) return 0.0;
    double raw;
    if (!readFileDouble(path, raw)) return 0.0;
    return raw / 1000.0;
}

}  // namespace

struct PerfLogger::Impl {
    Config cfg;
    PerfWindow windows[kMetricCount];
    // PerfWindow bir std::mutex barındırdığı için kopyalanamaz/taşınamaz —
    // std::vector<PerfWindow> resize()/assign() gibi işlemlerde (boş
    // vector'den büyütülse bile) şablon içinde move-constructible ARANIR,
    // bu yüzden derlenmez. unique_ptr sarmalayıcı bu kısıtı aşar (unique_ptr
    // taşınabilir, işaret ettiği PerfWindow hiç taşınmaz).
    std::vector<std::unique_ptr<PerfWindow>> core_windows;
    long core_count = 1;

    uint64_t frame_count = 0;
    std::mutex frame_count_mutex;

    NPUMonitor npu_monitor;
    PowerMonitor power_monitor;
    std::string thermal_path;

    // 500ms'de bir CPU örnekleme (ARM HZ=100 jiffy artifact önlemi —
    // perf_logger_v2.py ile aynı disiplin).
    double cpu_sample_interval_s = 0.5;
    uint64_t last_cpu_sample_us = 0;
    std::vector<CpuTimes> last_core_times;
    uint64_t last_process_jiffies = 0;
    std::vector<double> last_core_pcts;
    double last_process_cpu_pct = 0.0;
    long clk_tck = 100;

    std::chrono::steady_clock::time_point start_time;
    bool start_time_set = false;

    explicit Impl(Config c) : cfg(std::move(c)) {
        for (auto& w : windows) w.setMaxSize(cfg.window);

        core_count = sysconf(_SC_NPROCESSORS_ONLN);
        if (core_count < 1) core_count = 1;
        for (long i = 0; i < core_count; ++i) {
            core_windows.push_back(std::make_unique<PerfWindow>(cfg.window));
        }
        last_core_times.assign(static_cast<size_t>(core_count), CpuTimes{});
        last_core_pcts.assign(static_cast<size_t>(core_count), 0.0);

        clk_tck = sysconf(_SC_CLK_TCK);
        if (clk_tck < 1) clk_tck = 100;

        thermal_path = findThermalZone();

        // Warmup — ilk örnekleme referans noktası (perf_logger_v2.py'nin
        // psutil.cpu_percent(interval=None) ilk-çağrı-referans davranışıyla
        // aynı amaç).
        sampleCpu(true);

        initCsv();
    }

    void initCsv() {
        std::ofstream f(cfg.log_path, std::ios::trunc);
        f << "timestamp,frame_count";
        for (int m = 0; m < kMetricCount; ++m) {
            f << ',' << kMetricNames[m] << "_avg," << kMetricNames[m] << "_min,"
              << kMetricNames[m] << "_max";
        }
        for (long i = 0; i < core_count; ++i) {
            f << ",cpu_core_" << i << "_pct_avg,cpu_core_" << i << "_pct_min,cpu_core_"
              << i << "_pct_max";
        }
        f << '\n';
    }

    // force=true: sadece iç referans durumunu kurar, per-core/process
    // yüzdelerini HESAPLAMAZ (ilk çağrıda anlamlı delta yok).
    void sampleCpu(bool force_init = false) {
        uint64_t now_us = nowUs();
        double elapsed_s = (now_us - last_cpu_sample_us) / 1e6;
        if (!force_init && elapsed_s < cpu_sample_interval_s) return;

        std::ifstream stat("/proc/stat");
        if (stat) {
            std::string label;
            CpuTimes agg;
            readProcStatLine(stat, label, agg);  // ilk satır: toplam "cpu " (kullanılmıyor)
            for (long i = 0; i < core_count; ++i) {
                CpuTimes cur;
                if (!readProcStatLine(stat, label, cur)) break;
                if (!force_init) {
                    uint64_t d_total = cur.total > last_core_times[i].total
                                            ? cur.total - last_core_times[i].total
                                            : 0;
                    uint64_t d_idle = cur.idle_all > last_core_times[i].idle_all
                                           ? cur.idle_all - last_core_times[i].idle_all
                                           : 0;
                    double pct = (d_total > 0)
                                     ? (100.0 * (static_cast<double>(d_total) -
                                                 static_cast<double>(d_idle)) /
                                        static_cast<double>(d_total))
                                     : 0.0;
                    last_core_pcts[i] = std::max(0.0, std::min(100.0, pct));
                }
                last_core_times[i] = cur;
            }
        }

        uint64_t proc_jiffies = 0;
        if (readProcessJiffies(proc_jiffies)) {
            if (!force_init && elapsed_s > 0) {
                double d_jiffies = static_cast<double>(proc_jiffies > last_process_jiffies
                                                            ? proc_jiffies - last_process_jiffies
                                                            : 0);
                double proc_cpu_pct_total = (d_jiffies / clk_tck) / elapsed_s * 100.0;
                // psutil.Process.cpu_percent() ile aynı normalize: toplam
                // sistem kapasitesine göre pay (core_count'a böl) —
                // background_cpu_pct = cpu_system_pct - process_cpu_pct
                // farkının anlamlı olması için ikisi de AYNI ölçekte olmalı.
                last_process_cpu_pct = proc_cpu_pct_total / core_count;
            }
            last_process_jiffies = proc_jiffies;
        }

        last_cpu_sample_us = now_us;
    }
};

PerfLogger::PerfLogger(Config cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}
PerfLogger::~PerfLogger() { stop(); }

void PerfLogger::update(double frame_ms, double decode_ms, double invoke_ms,
                         double postproc_ms) {
    if (!impl_->start_time_set) {
        impl_->start_time = std::chrono::steady_clock::now();
        impl_->start_time_set = true;
    }

    double fps = frame_ms > 0 ? 1000.0 / frame_ms : 0.0;
    double ram_mb = readRssMb();
    double npu_pct = frame_ms > 0 ? (invoke_ms / frame_ms * 100.0) : 0.0;
    double temp_c = readTemperature(impl_->thermal_path);

    impl_->sampleCpu();
    double cpu_system_pct = 0.0;
    for (double p : impl_->last_core_pcts) cpu_system_pct += p;
    if (!impl_->last_core_pcts.empty()) cpu_system_pct /= impl_->last_core_pcts.size();
    double process_cpu_pct = impl_->last_process_cpu_pct;
    double background_cpu_pct = std::max(0.0, cpu_system_pct - process_cpu_pct);

    double npu_peaks[3];
    impl_->npu_monitor.getPeakAndReset(npu_peaks);
    double power_w = impl_->power_monitor.readWatts();

    double values[kMetricCount] = {
        fps, frame_ms, decode_ms, invoke_ms, postproc_ms,
        cpu_system_pct, process_cpu_pct, background_cpu_pct, cpu_system_pct,
        npu_pct, ram_mb, temp_c,
        npu_peaks[0], npu_peaks[1], npu_peaks[2], power_w,
    };
    for (int m = 0; m < kMetricCount; ++m) impl_->windows[m].update(values[m]);

    for (size_t i = 0; i < impl_->last_core_pcts.size(); ++i) {
        impl_->core_windows[i]->update(impl_->last_core_pcts[i]);
    }

    bool should_flush;
    {
        std::lock_guard<std::mutex> lk(impl_->frame_count_mutex);
        ++impl_->frame_count;
        should_flush = (impl_->frame_count % impl_->cfg.flush_every == 0);
    }
    if (should_flush) flush();
}

void PerfLogger::flush() {
    std::ofstream f(impl_->cfg.log_path, std::ios::app);
    f << isoTimestamp() << ',' << impl_->frame_count;
    for (int m = 0; m < kMetricCount; ++m) {
        auto s = impl_->windows[m].stats();
        f << ',' << std::fixed << std::setprecision(2) << s.avg << ',' << s.min << ','
          << s.max;
    }
    for (auto& cw : impl_->core_windows) {
        auto s = cw->stats();
        f << ',' << std::fixed << std::setprecision(2) << s.avg << ',' << s.min << ','
          << s.max;
    }
    f << '\n';
}

std::string PerfLogger::summary() const {
    auto fmt_row = [](const std::string& name, const PerfWindow::Stats& s,
                       const std::string& unit) {
        std::ostringstream oss;
        if (s.count == 0) {
            oss << "  " << std::left << std::setw(18) << name << std::right << std::setw(10)
                << "N/A" << std::setw(10) << "N/A" << std::setw(10) << "N/A";
            return oss.str();
        }
        oss << "  " << std::left << std::setw(18) << name << std::right << std::fixed
            << std::setprecision(1) << std::setw(8) << s.avg << unit << " " << std::setw(8)
            << s.min << unit << " " << std::setw(8) << s.max << unit;
        return oss.str();
    };

    std::ostringstream out;
    out << std::string(64, '=') << '\n';
    out << "  PERFORMANCE SUMMARY\n";
    out << std::string(64, '=') << '\n';
    out << "  " << std::left << std::setw(18) << "Metric" << std::right << std::setw(10)
        << "AVG" << std::setw(10) << "MIN" << std::setw(10) << "MAX" << '\n';
    out << "  " << std::string(60, '-') << '\n';

    // hız/gecikme + genel CPU/NPU/RAM/sicaklik (npu_core/power_w ve cpu_pct
    // alias'i haric — perf_logger_v2.py'nin summary()'siyle ayni disiplin).
    const int base_order[] = {kFps,        kFrameMs,       kDecodeMs,       kInvokeMs,
                               kPostprocMs, kCpuSystemPct,  kProcessCpuPct,  kBackgroundCpuPct,
                               kNpuPct,     kRamMb,         kTempC};
    for (int m : base_order) {
        out << fmt_row(kMetricNames[m], impl_->windows[m].stats(), kMetricUnits[m]) << '\n';
    }

    out << "  " << std::string(60, '-') << '\n';
    out << "  Per-Core CPU Usage\n";
    out << "  " << std::string(60, '-') << '\n';
    for (size_t i = 0; i < impl_->core_windows.size(); ++i) {
        out << fmt_row("cpu_core_" + std::to_string(i) + "_pct",
                        impl_->core_windows[i]->stats(), "%")
            << '\n';
    }

    out << "  " << std::string(60, '-') << '\n';
    out << "  Hardware Counters (RK3588)\n";
    out << "  " << std::string(60, '-') << '\n';
    const int hw_order[] = {kNpuCore0, kNpuCore1, kNpuCore2, kPowerW};
    for (int m : hw_order) {
        out << fmt_row(kMetricNames[m], impl_->windows[m].stats(), kMetricUnits[m]) << '\n';
    }

    double duration_s =
        impl_->start_time_set
            ? std::chrono::duration<double>(std::chrono::steady_clock::now() - impl_->start_time)
                  .count()
            : 0.0;
    int mins = static_cast<int>(duration_s) / 60;
    int secs = static_cast<int>(duration_s) % 60;

    out << "  " << std::string(60, '-') << '\n';
    out << "  Window size  : " << impl_->windows[kFps].stats().count << " frames\n";
    out << "  Total frames : " << impl_->frame_count << '\n';
    out << "  Duration     : " << mins << "m " << secs << "s\n";
    out << "  Logical cores: " << impl_->core_count << '\n';
    out << std::string(64, '=') << '\n';
    return out.str();
}

void PerfLogger::stop() { impl_->npu_monitor.stop(); }