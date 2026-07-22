#pragma once

#include <cstdint>
#include <memory>
#include <string>

// RK3588 Edge AI Workshop'un perf_logger_v2.py'siyle AYNI metrik seti ve
// CSV şeması — mLogger_test.cpp'nin logladığı .csv, Python tarafındaki
// analiz scriptleri/grafiklerle DOĞRUDAN uyumlu olsun diye kasıtlı olarak
// birebir kopyalandı (kolon adları, sliding-window istatistikleri (min/max/
// avg), flush aralığı, summary() formatı).
//
// EŞLEŞTİRME NOTU (Python'un GStreamer/OpenCV blocking-capture modeliyle
// bizim MPP tabanlı, TAMAMEN ASENKRON decode mimarimiz arasındaki fark):
// Python'da decode_ms = "appsink.emit('try-pull-sample')" çağrısının
// bloklama süresi — capture, invoke'dan HEMEN ÖNCE aynı thread'de olur.
// Bizim mimarimizde MPP decode kendi thread'lerinde (put/get loop) sürekli
// ve bağımsız akar (06-system-architecture.md §5) — tracking thread'i
// invoke'dan önce decode'u hiç beklemez, "en yeni kare" slotundan anlık
// alır. tracking thread'inin invoke'dan ÖNCE harcadığı gerçek "girdi
// hazırlama" süresi RGA crop'udur (ITracker::lastTiming().crop_us). Bu
// yüzden decode_ms kolonu isim uyumluluğu için (aynı CSV şeması) korunuyor
// ama İÇERİĞİ crop_us'tur — video kod çözme süresi DEĞİL (video kod çözme
// zaten tracking'i hiç bloklamıyor, bu da NanoTrack/LightTrack farkının
// asıl nedeniydi, bkz. main_m9_test.cpp'deki cloneFrame commit mesajı).
class PerfLogger {
public:
    struct Config {
        std::string log_path;
        uint32_t window;       // sliding window boyutu (kare)
        uint32_t flush_every;  // kaç karede bir CSV'ye satır yazılır

        // KASITLI: varsayılan değerler member initializer (NSDMI) yerine
        // burada, sıradan bir constructor gövdesinde verilir. Nested bir
        // sınıfın NSDMI'leri, "complete-class context"ı ancak ÇEVRELEYEN
        // sınıfın (PerfLogger) kapanışında tamamlandığından, PerfLogger'ın
        // KENDİ İÇİNDE "Config cfg = Config{}" gibi bir varsayılan argümanda
        // kullanılamıyor (GCC: "default member initializer ... required
        // before the end of its enclosing class"). Sıradan bir ctor bu
        // kısıta tabi değil.
        Config() : log_path("perf_metrics.csv"), window(300), flush_every(100) {}
    };

    explicit PerfLogger(Config cfg = Config{});
    ~PerfLogger();

    PerfLogger(const PerfLogger&) = delete;
    PerfLogger& operator=(const PerfLogger&) = delete;

    // perf_logger_v2.py'deki PerfLoggerV2.update(frame_ms, decode_ms,
    // invoke_ms, postproc_ms) ile birebir aynı imza/semantik. flush_every
    // karede bir otomatik flush() tetiklenir.
    void update(double frame_ms, double decode_ms, double invoke_ms, double postproc_ms);

    // O ana kadarki sliding window istatistiklerini CSV'ye bir satır olarak
    // yazar (timestamp, frame_count, her metrik için avg/min/max).
    void flush();

    // Konsola perf_logger_v2.py'nin summary()'siyle aynı formatta bir özet
    // tablosu döner (fps/frame/decode/invoke/postproc, CPU, NPU, RAM, sıcaklık,
    // per-core CPU, donanım sayaçları: npu_core0-2, power_w).
    std::string summary() const;

    // NPUMonitor'ın arka plan thread'ini durdurur — programdan çıkmadan
    // ÖNCE çağrılmalı (perf_logger_v2.py'nin "segfault önlemi" notuyla aynı
    // disiplin: thread'ler ana thread'den önce/karışık kapanmamalı).
    void stop();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};