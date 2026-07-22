#pragma once

#include "inference/RknnInferenceEngine.hpp"
#include "preprocess/RgaPreprocessor.hpp"
#include "tracking/ITracker.hpp"
#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <string>
#include <vector>

// LightTrack'e ÖZGÜ tüm hiperparametreler ve durum bu dosyada kapalı.
// NanoTrack (veya başka bir Siamese tracker) eklemek istendiğinde, bu
// dosyaya dokunmadan yeni bir NanoTrackImpl yazılır — ITracker/
// RknnInferenceEngine/RgaPreprocessor aynen kullanılır (kullanıcı isteği:
// "uyumluluğu hızlı sağlayalım").
//
// Değerler config.json'daki "tracking_parameters" bölümünden alınmıştır.
struct LightTrackConfig {
    int stride = 16;
    float penalty_k = 0.062f;
    float window_influence = 0.38f;
    float lr = 0.765f;
    int exemplar_size = 127;
    int instance_size = 288;
    int total_stride = 16;
    int score_size = 18;
    float context_amount = 0.5f;
    float lost_score_threshold = 0.3f;
    // Referans kod 5'e kadar destekliyor (base + flip + parlak + karanlık
    // + büyük-ölçek). Bu ilk versiyonda SADECE base+flip+büyük-ölçek (RGA
    // ile doğrudan karşılığı olan 3 augmentasyon) destekleniyor —
    // parlaklık/karanlık augmentasyonları RGA'nın imquantize fonksiyonuyla
    // teorik olarak mümkün ama semantiği (im_nn_t scale/offset'in NN
    // normalizasyonu için mi yoksa fotografik parlaklık için mi
    // tasarlandığı) header'dan doğrulanamadı — yanlış varsayımla sessizce
    // hatalı augmentasyon üretmektense atlandı. num_augments 3'ü aşan
    // değerler bu implementasyonda 3'e sabitlenir.
    int num_augments = 5;
};

struct LightTrackState {
    int im_w = 0;
    int im_h = 0;
    // Kırpma sınır-dışı dolgu rengi. Referans cv::mean(img) ile gerçek
    // ortalamayı hesaplıyordu; bu ilk versiyonda YOLO letterbox'ıyla
    // tutarlı sabit nötr gri kullanılıyor (Rapor 3 — 114,114,114) çünkü
    // bu değer sadece hedef kare kenarına yakınken görünen dolgu
    // pikselleri için kozmetik önem taşıyor, izleme doğruluğunu
    // belirlemiyor.
    uint8_t pad_r = 114, pad_g = 114, pad_b = 114;

    float target_cx = 0.f, target_cy = 0.f;
    float target_w = 0.f, target_h = 0.f;
    float last_good_w = 0.f, last_good_h = 0.f;
    float cls_score_max = 0.f;
};

class LightTrackImpl : public ITracker {
public:
    LightTrackImpl();
    ~LightTrackImpl() override;

    // template_model/search_model/head_model: sırasıyla lighttrack_init_*,
    // lighttrack_backbone_*, lighttrack_neck_head_* .rknn dosyaları.
    //
    // NPU çekirdek ataması OTOMATİK (RKNN_NPU_CORE_AUTO).
    // Referans LightTrack projesindeki (RKEngine) ile aynı davranış.
    bool loadModels(const std::string& template_model, const std::string& search_model,
                     const std::string& head_model);

    void setConfig(const LightTrackConfig& cfg);

    bool initialize(const DmaBufferPtr& frame, const BBox& bbox) override;
    TrackResult track(const DmaBufferPtr& frame) override;
    TrackerState getState() const override;
    TrackTiming lastTiming() const override;
    void shutdown() override;

private:
    void createWindow();
    void createGrids();
    // updateFromSearchCrop: lighttrack.cpp'deki update()'in birebir portu
    // (transpose, sigmoid, penalty, argmax, bbox decode) — state_ üyelerini
    // doğrudan günceller.
    void updateFromSearchCrop(const DmaBufferPtr& search_crop, float scale_z);

    RknnInferenceEngine template_engine_;  // t_engine_ karşılığı (init)
    RknnInferenceEngine search_engine_;    // x_engine_ karşılığı (backbone)
    RknnInferenceEngine head_engine_;      // h_engine_ karşılığı (neck_head)
    RgaPreprocessor preprocessor_;

    LightTrackConfig cfg_;
    LightTrackState state_;
    TrackerState tracker_state_ = TrackerState::Idle;

    std::vector<float> window_;
    std::vector<float> grid_x_;
    std::vector<float> grid_y_;

    // TEŞHİS: ilk birkaç karede cls skor dağılımını (min/max/ortalama),
    // peak konumunu ve hesaplanan yer değiştirmeyi loglar — "skorlar düz
    // gürültü mü, peak Hanning merkezine mi yapışıyor" sorusunu canlı
    // koşuda tek bakışta cevaplamak için (M6 donmuş-bbox teşhisi).
    int debug_frames_logged_ = 0;

    // PERFORMANS TEŞHİSİ: aşama bazlı süre toplamları (mikrosaniye).
    // Her 60 track() çağrısında ortalamalar loglanıp sıfırlanır — kartta
    // FPS darboğazının hangi aşamada olduğunu (RGA crop mu, backbone NPU
    // mu, head NPU mu, CPU post mu) kesin sayıyla gösterir.
    uint64_t perf_crop_us_ = 0;
    uint64_t perf_backbone_us_ = 0;
    uint64_t perf_head_us_ = 0;
    uint64_t perf_post_us_ = 0;
    // NPU'nun KENDİ bildirdiği saf inference süreleri (RKNN_QUERY_PERF_RUN)
    // — duvar süresi ile arasındaki fark, rknn_inputs_set'in CPU dönüşümü
    // (UINT8->FP16 normalize) + want_float dequant maliyetidir. "Backbone
    // 36ms" gibi bir duvar süresinin NPU mu CPU mu olduğunu bu ayırır.
    uint64_t perf_backbone_npu_us_ = 0;
    uint64_t perf_head_npu_us_ = 0;
    int perf_frames_ = 0;

    // Son BAŞARILI kare için (60-karelik ortalamaların aksine) TEK karelik
    // değerler — lastTiming()/mLogger_test.cpp bunları okur.
    uint64_t last_crop_us_ = 0;
    uint64_t last_backbone_us_ = 0;
    uint64_t last_head_us_ = 0;
    uint64_t last_post_us_ = 0;

    // head_engine_ girişleri için NCHW->NHWC transpoze edilmiş buffer'lar.
    // zf_transposed_: template_engine_ çıktısının transpozu, initialize()'da
    // BİR KEZ hesaplanır ve track() boyunca CACHE'lenir (sabit şablon).
    std::vector<float> zf_transposed_;
    // xf_transposed_: search_engine_ çıktısının transpozu, HER KAREDE
    // yeniden hesaplanır.
    std::vector<float> xf_transposed_;

    bool models_loaded_ = false;
};