// M6 doğrulama: RTSP demux (M1) + native MPP decode (M2) + LightTrack
// (M5'in 3 motoruyla, gerçek üretim çekirdek maskeleriyle) + RGA ile
// doğrudan kareye bbox çizimi + RTSP yeniden yayın (M3/M4).
//
// Amaç: tracking'in GERÇEK ZAMANLI bir RTSP akışı üzerinde, referans
// uygulamayla (lighttrack-rk3588) aynı matematikle çalıştığını GÖZLE
// kanıtlamak. IBufferPool ve PipelineOrchestrator BİLEREK henüz yok —
// M1-M5 ile aynı disiplin: önce doğruluk (tek-thread, kare-başına tek
// seferlik tahsis), havuzlama/orkestrasyon doğruluk kanıtlandıktan
// sonraki bir adım (bkz. CLAUDE.md "Modül Durumu").
//
// Akış: ilk kare geldiğinde CLI'dan verilen bbox ile tracker.initialize()
// çağrılır (referans uygulamanın --headless --bbox moduyla birebir aynı
// varsayım: ilk hedef konumu elle verilir, detector henüz yok). Sonraki
// her karede tracker.track() çağrılır, sonuç bbox'ı RGA ile (imfill, 4
// ince şerit — immakeBorder DEĞİL, bkz. 01-IPreprocessor.md §3.1) doğrudan
// decode çıktısının üzerine çizilir, ardından kare RTSP üzerinden yeniden
// yayınlanır.
//
// KRİTİK: track()/initialize() çağrısı MppDecoder'ın kendi get_thread'i
// üzerinde, SERİ olarak çalışır (05-ITracker.md §9 kuralıyla doğal olarak
// uyumlu — ayrı bir tracking thread'i kurmaya GEREK YOK, decoder zaten
// tek ve seri bir thread veriyor).
//
// Derleme (M7'den itibaren src/buffer/DmaBufferPool.cpp de gerekli —
// RgaPreprocessor buffer'larını artık IBufferPool'dan alıyor):
//   g++ -std=c++14 -O2 \
//     src/ingestion/RtspDemuxer.cpp src/decode/MppDecoder.cpp \
//     src/buffer/DmaBufferPool.cpp \
//     src/preprocess/RgaPreprocessor.cpp src/inference/RknnInferenceEngine.cpp \
//     src/tracking/LightTrackImpl.cpp src/streaming/RtspStreamer.cpp \
//     src/tracking/main_m6_test.cpp \
//     -Iinclude -Ilibrknn_api/include -I/usr/include/rockchip -I/usr/include/rga \
//     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 \
//        gstreamer-rtsp-server-1.0) \
//     -Llibrknn_api/aarch64 -lrknnrt -Wl,-rpath,librknn_api/aarch64 \
//     -lrockchip_mpp -lrga -lpthread -o m6_tracking_test
//
// Çalıştırma (kartta):
//   ./m6_tracking_test rtsp://<kaynak_ip>:8554/stream 233,152,109,222 8557
//
// İzleme (PC'de, ayrı bir terminalde):
//   ffplay rtsp://<kart_ip>:8557/out

#include "decode/MppDecoder.hpp"
#include "ingestion/RtspDemuxer.hpp"
#include "streaming/RtspStreamer.hpp"
#include "tracking/LightTrackImpl.hpp"

#include "im2d.h"
#include "rga.h"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_running{true};
void onSigint(int) { g_running.store(false); }

// bounding_box_parameters (config.json) ile AYNI eşikler. LightTrackConfig
// da aynı sebeple (henüz bir JSON okuyucu yok) varsayılanlarını elle
// config.json ile senkron tutuyor — burada da aynı disiplin uygulanıyor.
constexpr float kScoreGreenThreshold = 0.7f;
constexpr float kScoreOrangeThreshold = 0.4f;
constexpr int kBoxThickness = 4;

int toRgaFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::NV12:
            return RK_FORMAT_YCbCr_420_SP;
        case PixelFormat::NV16:
            return RK_FORMAT_YCbCr_422_SP;
        case PixelFormat::RGB888:
            return RK_FORMAT_RGB_888;
        case PixelFormat::BGR888:
            return RK_FORMAT_BGR_888;
        default:
            return RK_FORMAT_YCbCr_420_SP;
    }
}

// RGA imfill'i 4 kez çağırarak dikdörtgen KENARLIĞI oluşturur (dolu kutu
// değil, çerçeve). immakeBorder KULLANILMIYOR (RGA2 + DMA32/IOMMU çökme
// riski, kesin yasak — bkz. 01-IPreprocessor.md §3.1); bu sadece imfill'in
// doğrudan tekrarı, RgaPreprocessor'ın letterbox zemin doldurmasıyla AYNI
// donanımsal çağrı. CPU piksele dokunmaz — decode çıktısının fd'si
// doğrudan RGA'ya wrap edilip üzerine yazılır.
//
// NOT (RgaPreprocessor.cpp'deki gri-dolgu notuyla AYNI belirsizlik):
// color parametresinin NV12/NV16 hedefte gerçekte nasıl yorumlandığı
// header'da belirtilmemiş — RGB paketlenmiş int deneniyor, sonucu gözle
// doğrulayıp gerekirse düzelteceğiz.
void drawBboxOutline(const DmaBufferPtr& frame, const BBox& bbox, int color) {
    static bool logged_error = false;
    if (!frame || frame->fd < 0) return;

    int fw = static_cast<int>(frame->width);
    int fh = static_cast<int>(frame->height);

    int x0 = std::max(0, bbox.x);
    int y0 = std::max(0, bbox.y);
    int x1 = std::min(fw, bbox.x + bbox.width);
    int y1 = std::min(fh, bbox.y + bbox.height);
    if (x1 <= x0 || y1 <= y0) return;

    int t = std::min(kBoxThickness, std::min(x1 - x0, y1 - y0) / 2);
    if (t < 1) t = 1;

    rga_buffer_t buf = wrapbuffer_fd(frame->fd, fw, fh, toRgaFormat(frame->format),
                                      static_cast<int>(frame->w_stride),
                                      static_cast<int>(frame->h_stride));

    // RgaPreprocessor::cropTargetCentric'te canlı testte keşfedilen AYNI
    // kısıt: NV12/NV16 (chroma alt-örnekleme) hedeflerde imfill dikdörtgeni
    // de x/y/genişlik/yükseklik'te 2'ye hizalı olmak zorunda, aksi halde
    // sessizce reddediliyor. fw/fh (video çözünürlüğü) zaten çift olduğundan
    // x/y'yi aşağı, genişlik/yüksekliği aşağı 2'ye yuvarlamak (min 2)
    // sınırlar içinde kalmayı garantiler.
    auto alignRectEven = [](im_rect r) {
        r.x -= (r.x & 1);
        r.y -= (r.y & 1);
        r.width -= (r.width & 1);
        r.height -= (r.height & 1);
        if (r.width < 2) r.width = 2;
        if (r.height < 2) r.height = 2;
        return r;
    };

    // 4 kenar TEK RGA işinde (imfillArray) — 4 ayrı imfill çağrısı 4 ayrı
    // donanım işi/senkronizasyonu demekti (kare başına gereksiz ~ms'ler).
    im_rect rects[4] = {
        alignRectEven(im_rect{x0, y0, x1 - x0, t}),
        alignRectEven(im_rect{x0, std::max(y0, y1 - t), x1 - x0, t}),
        alignRectEven(im_rect{x0, y0, t, y1 - y0}),
        alignRectEven(im_rect{std::max(x0, x1 - t), y0, t, y1 - y0}),
    };

    IM_STATUS r = imfillArray(buf, rects, 4, static_cast<uint32_t>(color));

    if (!logged_error && r != IM_STATUS_SUCCESS) {
        std::cerr << "[M6] drawBboxOutline: imfillArray başarısız: " << imStrError(r)
                  << " (bir kez loglanıyor)\n";
        logged_error = true;
    }
}

int colorForConfidence(float confidence, bool lost) {
    if (lost || confidence <= kScoreOrangeThreshold) {
        return (255 << 16) | (0 << 8) | 0;  // kırmızı — kayıp/düşük güven
    }
    if (confidence <= kScoreGreenThreshold) {
        return (255 << 16) | (165 << 8) | 0;  // turuncu — orta güven
    }
    return (0 << 16) | (255 << 8) | 0;  // yeşil — güvenilir takip
}

// TEŞHİS AMAÇLI (kalıcı özellik değil): RgaPreprocessor::cropTargetCentric'in
// ürettiği NV12/NV16 -> RGB888 dönüşümünü gözle doğrulamak için ham crop'u
// bir .ppm dosyasına yazar. LightTrack matematiği referansla birebir
// örtüşmesine rağmen bbox hiç hareket etmiyorsa (canlı testte gözlemlendi),
// en olası neden ağa giden pikselin bozuk olması — bu, projede improcess'in
// GERÇEKTEN YUV->RGB renk dönüşümü yaptığı ilk kod yolu (M3 sadece
// NV16->NV12 denemişti, hiç RGB'ye çevirmemişti). Bu dump, "gerçek bir
// görüntü mü yoksa renk çorbası mı" sorusunu saniyeler içinde cevaplar.
void dumpRgbCropAsPpm(const DmaBufferPtr& crop, const std::string& path) {
    if (!crop || !crop->virt_addr) {
        std::cerr << "[M6] dumpRgbCropAsPpm: crop boş, dump atlandı\n";
        return;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[M6] dumpRgbCropAsPpm: " << path << " açılamadı\n";
        return;
    }
    f << "P6\n" << crop->width << " " << crop->height << "\n255\n";
    f.write(static_cast<const char*>(crop->virt_addr),
            static_cast<std::streamsize>(crop->width) * crop->height * 3);
    std::cout << "[M6] Teşhis crop'u yazıldı: " << path << " (" << crop->width << "x"
              << crop->height << ", scp ile indirip bir görüntü görüntüleyiciyle açın)\n";
}

// Tracking thread'ini RK3588'in BÜYÜK çekirdeklerine (A76, CPU 4-7) sabitler.
// rknn_inputs_set'in UINT8->FP16 dönüşümü (kare başına 243KB, backbone duvar
// süresindeki ~14ms'lik CPU payının ana bileşeni) bu thread'de koşuyor —
// zamanlayıcı thread'i küçük A55 çekirdeklerine düşürürse aynı iş 3-4x
// yavaşlıyor. Başarısızlık ölümcül değil (sadece yavaş), o yüzden hata
// durumunda uyarıp devam ediyoruz.
void pinToBigCores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 4; c <= 7; ++c) CPU_SET(c, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        std::cerr << "[M6] UYARI: tracking thread'i büyük çekirdeklere "
                     "sabitlenemedi (devam ediliyor)\n";
    }
}

// Affinity'yi tüm çekirdeklere geri açar. KULLANIM SIRASI ÖNEMLİ (canlı
// ölçümden çıkan ders): tracking thread'ini pinlemek head tarafındaki CPU
// payını yarıya indirdi ama backbone'un ~13ms'lik CPU payına DOKUNMADI —
// çünkü RKNN runtime'ın iç işçi thread'leri (girdi dönüşümünü yapanlar)
// rknn_init sırasında, yani ANA thread'de yaratılıyor ve affinity'yi o
// andaki ana thread'den miras alıyor. Bu yüzden main(), loadModels'ten
// ÖNCE pinToBigCores() çağırır (işçiler A76 miras alır), model yüklemesi
// bitince bununla affinity'yi geri açar (decoder/demuxer/streamer
// thread'leri tüm çekirdeklere dağılabilsin).
void restoreAllCores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    long n = sysconf(_SC_NPROCESSORS_CONF);
    if (n < 1) n = 8;
    for (long c = 0; c < n; ++c) CPU_SET(static_cast<int>(c), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// NPU devfreq durumunu okuyup basar. NPU'nun 'ondemand' governor'la düşük
// frekansta (300MHz taban, 1GHz maks) takılı kalması, backbone'un NPU
// süresinin (~22ms ölçüldü) beklenenin ~3 katına çıkmasını açıklayan en
// olası neden — bunu tahmin değil, ölçümle görünür kılıyoruz.
void printNpuDevfreq() {
    std::ifstream gov_f("/sys/class/devfreq/fdab0000.npu/governor");
    std::ifstream freq_f("/sys/class/devfreq/fdab0000.npu/cur_freq");
    std::string gov, freq;
    if ((gov_f >> gov) && (freq_f >> freq)) {
        std::cout << "[M6] NPU devfreq: governor=" << gov << " cur_freq=" << freq
                  << " Hz\n";
        if (gov != "performance") {
            std::cout << "[M6] ÖNERİ: NPU'yu sabit tam frekansa almak için:\n"
                         "      sudo sh -c 'echo performance > "
                         "/sys/class/devfreq/fdab0000.npu/governor'\n";
        }
    } else {
        std::cout << "[M6] NPU devfreq okunamadı (yol bu kernelde farklı olabilir: "
                     "ls /sys/class/devfreq/ ile bakın)\n";
    }
}

bool parseBbox(const std::string& s, BBox& out) {
    std::stringstream ss(s);
    std::string tok;
    int vals[4];
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try {
            vals[i] = std::stoi(tok);
        } catch (...) {
            return false;
        }
    }
    out.x = vals[0];
    out.y = vals[1];
    out.width = vals[2];
    out.height = vals[3];
    return out.width > 0 && out.height > 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Kullanım: " << argv[0]
                  << " rtsp://kaynak/yol x,y,w,h [cikis_port=8557] [width=1280] "
                     "[height=720] [template.rknn] [search.rknn] [head.rknn]\n"
                  << "Örnek: " << argv[0]
                  << " rtsp://192.168.1.10:8554/stream 233,152,109,222 8557\n";
        return 1;
    }

    std::string rtsp_url = argv[1];
    BBox init_bbox;
    if (!parseBbox(argv[2], init_bbox)) {
        std::cerr << "[M6] Geçersiz bbox formatı, beklenen: x,y,w,h (pozitif w/h)\n";
        return 1;
    }

    uint16_t out_port = (argc >= 4) ? static_cast<uint16_t>(std::stoi(argv[3])) : 8557;
    uint32_t out_w = (argc >= 5) ? static_cast<uint32_t>(std::stoi(argv[4])) : 1280;
    uint32_t out_h = (argc >= 6) ? static_cast<uint32_t>(std::stoi(argv[5])) : 720;
    std::string template_model =
        (argc >= 7) ? argv[6] : "models/lighttrack_init_fp16.rknn";
    std::string search_model =
        (argc >= 8) ? argv[7] : "models/lighttrack_backbone_fp16.rknn";
    std::string head_model =
        (argc >= 9) ? argv[8] : "models/lighttrack_neck_head_fp16.rknn";

    std::signal(SIGINT, onSigint);

    // KRİTİK SIRA (bkz. restoreAllCores yorumu): RKNN runtime'ın iç işçi
    // thread'leri rknn_init sırasında yaratılıp affinity'yi ana thread'den
    // miras aldığından, ana thread model yüklemesinden ÖNCE büyük
    // çekirdeklere pinlenir, yükleme bitince geri açılır.
    pinToBigCores();

    LightTrackImpl tracker;
    // setConfig() BİLEREK çağrılmıyor — LightTrackConfig varsayılanları
    // zaten config.json'un tracking_parameters bölümüyle senkron (bkz.
    // LightTrackImpl.hpp yorumu).
    if (!tracker.loadModels(template_model, search_model, head_model)) {
        std::cerr << "[M6] Tracker modelleri yüklenemedi: " << template_model << ", "
                  << search_model << ", " << head_model << "\n";
        return 1;
    }

    restoreAllCores();
    printNpuDevfreq();

    RtspDemuxer demuxer(rtsp_url);
    MppDecoder decoder(VideoCoding::H264);
    RtspStreamer streamer(out_port, "/out", out_w, out_h, 30);

    std::atomic<bool> initialized{false};
    std::atomic<uint64_t> frames_tracked{0};
    std::atomic<uint64_t> frames_lost{0};
    std::atomic<float> last_confidence{0.0f};

    // ================= DECODE / TRACKING AYRIŞTIRMASI =================
    // KRİTİK MİMARİ DÜZELTME (canlı perf ölçümünden sonra): track+çizim+
    // push (~45-50ms) decoder'ın frame callback'inde SERİ çalışınca decoder
    // yeni kare çekemiyordu -> decode 30'dan ~15 FPS'e düşüyordu; dahası
    // decoder yavaş tüketince MPP paket kuyruğu dolup demuxer paket
    // düşürüyor, bozulan H264 bir sonraki keyframe'e kadar çözülemiyor ->
    // loglardaki GOP-periyotlu "0 FPS" delikleri. Referans uygulamanın
    // (lighttrack_test.cpp, DropFrameQueue maxsize=2) çözdüğü problem
    // birebir buydu. Aynı desen: decode callback kareyi sadece "en yeni
    // kare" slotuna bırakır (eskisini düşürür), AYRI TEK bir tracking
    // thread'i kendi hızında tüketir — ITracker'ın tek-seri-thread kuralı
    // (05-ITracker.md §9) korunur, decoder asla bloklanmaz.
    std::mutex frame_mtx;
    std::condition_variable frame_cv;
    DmaBufferPtr latest_frame;
    std::atomic<uint64_t> frames_dropped{0};

    decoder.setFrameCallback([&](DmaBufferPtr frame) {
        {
            std::lock_guard<std::mutex> lk(frame_mtx);
            if (latest_frame) {
                frames_dropped.fetch_add(1, std::memory_order_relaxed);
            }
            latest_frame = std::move(frame);
        }
        frame_cv.notify_one();
    });

    auto processFrame = [&](const DmaBufferPtr& frame) {
        if (!initialized.load()) {
            std::cout << "[M6] İlk kare alındı (" << frame->width << "x" << frame->height
                      << "), tracker başlatılıyor, bbox=(" << init_bbox.x << ","
                      << init_bbox.y << "," << init_bbox.width << ","
                      << init_bbox.height << ")\n";
            if (!tracker.initialize(frame, init_bbox)) {
                std::cerr << "[M6] tracker.initialize() başarısız, akış durduruluyor\n";
                g_running.store(false);
                return;
            }
            initialized.store(true);

            // TEŞHİS: tracker.initialize()'ın kendi içindeki crop'a doğrudan
            // erişimimiz yok (LightTrackImpl bunu dışa vermiyor), o yüzden
            // AYNI bbox merkezi üzerinden bağımsız bir crop daha üretip
            // diske yazıyoruz — RGA'nın NV16->RGB dönüşümünün gerçekten
            // doğru çalışıp çalışmadığını gözle kontrol etmek için (bkz.
            // dumpRgbCropAsPpm yorumu).
            //
            // İKİ AYRI model_size İLE dump alıyoruz: 256 (=16x16, 16-hizalı,
            // REPACK YOLUNU TETİKLEMEZ — RgaPreprocessor.cpp'deki
            // needs_repack koşuluna bkz.) ve 127 (=exemplar_size, 16-hizalı
            // DEĞİL, tam olarak template crop'ların kullandığı REPACK
            // yolunu tetikler). İlk turda sadece 256'yı denemiştik ve
            // "düzgün" göründü — ama bu, repack kodunu hiç EGZERSİZ ETMEMİŞ
            // demekti. Şablon (zf) TAM OLARAK 127'lik crop'lardan üretiliyor;
            // 127'lik dump bozuksa (renk kayması, çapraz yırtılma vb.),
            // sorun kesin olarak RgaPreprocessor::cropTargetCentric'in
            // repack (stride->sıkı paket) kopyalama mantığında demektir.
            {
                RgaPreprocessor debug_pp;
                float cx = init_bbox.x + (init_bbox.width - 1) / 2.0f;
                float cy = init_bbox.y + (init_bbox.height - 1) / 2.0f;
                int ctx = std::max(init_bbox.width, init_bbox.height) * 3;
                RgaPixelFormat src_fmt = (frame->format == PixelFormat::NV16)
                                             ? RgaPixelFormat::NV16
                                             : RgaPixelFormat::NV12;

                DmaBufferPtr debug_crop_256;
                if (debug_pp.cropTargetCentric(frame, src_fmt, cx, cy, ctx, 256, 114, 114,
                                                114, debug_crop_256)) {
                    dumpRgbCropAsPpm(debug_crop_256, "/tmp/m6_debug_crop_256.ppm");
                }

                DmaBufferPtr debug_crop_127;
                if (debug_pp.cropTargetCentric(frame, src_fmt, cx, cy, ctx, 127, 114, 114,
                                                114, debug_crop_127)) {
                    dumpRgbCropAsPpm(debug_crop_127, "/tmp/m6_debug_crop_127.ppm");
                }
            }

            drawBboxOutline(frame, init_bbox, colorForConfidence(1.0f, false));
            streamer.pushFrame(frame);
            return;
        }

        TrackResult result = tracker.track(frame);
        bool lost = (tracker.getState() == TrackerState::Lost);
        frames_tracked.fetch_add(1, std::memory_order_relaxed);
        last_confidence.store(result.confidence, std::memory_order_relaxed);
        if (lost) {
            frames_lost.fetch_add(1, std::memory_order_relaxed);
        }

        drawBboxOutline(frame, result.bbox, colorForConfidence(result.confidence, lost));
        streamer.pushFrame(frame);
    };

    // Tracking thread'i — sistemdeki TEK track() çağıranı (seri kural).
    std::thread track_thread([&] {
        pinToBigCores();
        while (g_running.load()) {
            DmaBufferPtr frame;
            {
                std::unique_lock<std::mutex> lk(frame_mtx);
                frame_cv.wait_for(lk, std::chrono::milliseconds(100),
                                  [&] { return latest_frame != nullptr || !g_running.load(); });
                if (!latest_frame) continue;
                frame = std::move(latest_frame);
            }
            processFrame(frame);
        }
    });

    demuxer.setPacketCallback(
        [&](EncodedPacket&& pkt) { decoder.feedPacket(std::move(pkt)); });

    if (!streamer.start()) {
        std::cerr << "[M6] RTSP streamer başlatılamadı\n";
        return 1;
    }
    if (!decoder.start()) {
        std::cerr << "[M6] MPP decoder başlatılamadı\n";
        return 1;
    }
    if (!demuxer.start()) {
        std::cerr << "[M6] Demuxer başlatılamadı\n";
        return 1;
    }

    std::cout << "[M6] Zincir çalışıyor (kaynak -> decode -> LightTrack -> bbox çizimi "
                 "-> yeniden yayın), Ctrl+C ile durdurun...\n";

    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_tracked = 0;
    uint64_t last_decoded = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();
        if (elapsed >= 2.0) {
            uint64_t tf = frames_tracked.load(std::memory_order_relaxed);
            uint64_t dt = tf - last_tracked;
            // Decode FPS'i de raporla: "tracking FPS düşük" iki farklı şey
            // olabilir — pipeline yavaş (decode >> tracking) ya da KAYNAK
            // yayının kendisi kesik geliyor (decode da düşük/patlamalı).
            // Bu ayrımı yapmadan optimizasyon kararı verilemez.
            uint64_t df = decoder.getStats().frames_decoded;
            uint64_t dd = df - last_decoded;
            std::cout << "[M6] " << (dt / elapsed) << " FPS tracking, "
                      << (dd / elapsed) << " FPS decode, durum="
                      << (tracker.getState() == TrackerState::Lost ? "LOST" : "TRACKING")
                      << ", son confidence="
                      << last_confidence.load(std::memory_order_relaxed)
                      << ", düşürülen kare: "
                      << frames_dropped.load(std::memory_order_relaxed)
                      << ", toplam kayıp kare: "
                      << frames_lost.load(std::memory_order_relaxed) << "\n";
            last_report = now;
            last_tracked = tf;
            last_decoded = df;
        }
    }

    g_running.store(false);
    frame_cv.notify_all();
    if (track_thread.joinable()) track_thread.join();

    demuxer.stop();
    decoder.stop();
    streamer.stop();
    tracker.shutdown();
    std::cout << "[M6] Durduruldu.\n";
    return 0;
}
