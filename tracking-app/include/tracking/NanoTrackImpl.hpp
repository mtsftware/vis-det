#pragma once

#include "inference/RknnInferenceEngine.hpp"
#include "preprocess/RgaPreprocessor.hpp"
#include "tracking/ITracker.hpp"
#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <string>
#include <vector>

// NanoTrack (INT8) — LightTrackImpl'in yanına eklenen ikinci somut ITracker.
// LightTrackImpl.hpp'nin başındaki tasarım kararı bu dosyanın var oluş
// sebebi: "NanoTrack eklemek istendiğinde, bu dosyaya dokunmadan yeni bir
// NanoTrackImpl yazılır — ITracker/RknnInferenceEngine/RgaPreprocessor
// aynen kullanılır." Algoritma (grid/window/penalty/argmax/bbox decode)
// LightTrack ile birebir aynı aile (anchor-free Siamese) olduğundan bu
// dosya byte-byte LightTrackImpl'in yapısını izler; farklılık sadece
// hiperparametreler ve varsayılan model yolları.
//
// INT8 DEQUANTIZATION NOTU: referans nanotrack-rk3588/src/task/nanotrack.cpp
// çıktı tipini (FLOAT16 mı INT8 mi) kendisi sorgulayıp want_float'ı duruma
// göre açıyor, INT8 kaldığında deqnt_affine_to_f32(qnt,zp,scale) ile ELLE
// dequant ediyordu. Bizim RknnInferenceEngine'imiz bunu zaten GEREKSİZ
// kılıyor: getOutput() precision ne olursa olsun want_float=1 zorluyor
// (RKNN SDK'nın kendi affine-dequant'ını kullanıyor) — bkz.
// RknnInferenceEngine.hpp "want_float HER ZAMAN zorlanır" notu. Bu yüzden
// bu dosyada TensorProfile::zp/scale'e hiç dokunulmuyor; LightTrackImpl'deki
// gibi çıktılar doğrudan float* olarak okunuyor.
struct NanoTrackConfig {
    int stride = 16;
    float penalty_k = 0.138f;
    float window_influence = 0.435f;
    float lr = 0.348f;
    int exemplar_size = 127;
    int instance_size = 255;
    int total_stride = 16;
    int score_size = 16;
    float context_amount = 0.5f;
    // ITracker sözleşmesi getState()'in Lost'u raporlayabilmesini şart
    // koşuyor (LightTrackImpl'deki "kayıp hedef güvenlik ağı" ile aynı
    // rol) — referans nanotrack-rk3588/task bu kavramı hiç içermiyordu
    // (sadece cls_score_max döndürüyordu), bu alan SADECE bizim
    // ITracker/PipelineOrchestrator arayüzümüzün gereksinimi.
    float lost_score_threshold = 0.3f;
    // LightTrackImpl'deki aynı RGA stride kısıtı burada da geçerli
    // (exemplar_size=127, 127%16!=0) — flip CPU'da yapılıyor, bu yüzden
    // augmentasyon üst sınırı aynı sebeple 3'te tutuluyor (bkz.
    // LightTrackConfig::num_augments yorumu).
    int num_augments = 3;
};

struct NanoTrackState {
    int im_w = 0;
    int im_h = 0;
    uint8_t pad_r = 114, pad_g = 114, pad_b = 114;

    float target_cx = 0.f, target_cy = 0.f;
    float target_w = 0.f, target_h = 0.f;
    float last_good_w = 0.f, last_good_h = 0.f;
    float cls_score_max = 0.f;
};

class NanoTrackImpl : public ITracker {
public:
    NanoTrackImpl();
    ~NanoTrackImpl() override;

    // template_model/search_model/head_model: sırasıyla T_model_backbone,
    // X_model_backbone, model_head .rknn dosyaları (nanotrack-rk3588 ile
    // aynı adlandırma).
    bool loadModels(const std::string& template_model, const std::string& search_model,
                     const std::string& head_model);

    void setConfig(const NanoTrackConfig& cfg);

    bool initialize(const DmaBufferPtr& frame, const BBox& bbox) override;
    TrackResult track(const DmaBufferPtr& frame) override;
    TrackerState getState() const override;
    TrackTiming lastTiming() const override;
    void shutdown() override;

private:
    void createWindow();
    void createGrids();
    void updateFromSearchCrop(const DmaBufferPtr& search_crop, float scale_z);

    RknnInferenceEngine template_engine_;  // t_engine_ karşılığı
    RknnInferenceEngine search_engine_;    // x_engine_ karşılığı
    RknnInferenceEngine head_engine_;      // h_engine_ karşılığı
    RgaPreprocessor preprocessor_;

    NanoTrackConfig cfg_;
    NanoTrackState state_;
    TrackerState tracker_state_ = TrackerState::Idle;

    std::vector<float> window_;
    std::vector<float> grid_x_;
    std::vector<float> grid_y_;

    int debug_frames_logged_ = 0;

    uint64_t perf_crop_us_ = 0;
    uint64_t perf_backbone_us_ = 0;
    uint64_t perf_head_us_ = 0;
    uint64_t perf_post_us_ = 0;
    uint64_t perf_backbone_npu_us_ = 0;
    uint64_t perf_head_npu_us_ = 0;
    int perf_frames_ = 0;

    // Son BAŞARILI kare için TEK karelik değerler — lastTiming()/
    // mLogger_test.cpp bunları okur (bkz. LightTrackImpl aynı desen).
    uint64_t last_crop_us_ = 0;
    uint64_t last_backbone_us_ = 0;
    uint64_t last_head_us_ = 0;
    uint64_t last_post_us_ = 0;

    std::vector<float> zf_transposed_;
    std::vector<float> xf_transposed_;

    bool models_loaded_ = false;
};