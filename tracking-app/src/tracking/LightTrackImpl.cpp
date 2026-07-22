#include "tracking/LightTrackImpl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {
inline uint64_t nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}
}  // namespace

namespace {

// lighttrack.cpp ile BİREBİR AYNI hızlı sigmoid (fast_exp tabanlı).
inline float fastExp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = static_cast<uint32_t>((1 << 23) * (1.4426950409f * x + 126.93490512f));
    return v.f;
}
inline float sigmoidFast(float x) { return 1.0f / (1.0f + fastExp(-x)); }

float szWhFun(float w, float h) {
    float pad = (w + h) * 0.5f;
    return std::sqrt((w + pad) * (h + pad));
}

// NCHW(1,C,H,W) float -> NHWC(1,H,W,C) float. head_engine_ girişi bu
// düzeni bekliyor (lighttrack.cpp'nin transpose_nchw_to_nhwc'siyle aynı).
void transposeNchwToNhwc(const float* src, float* dst, int C, int H, int W) {
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                dst[h * (W * C) + w * C + c] = src[c * (H * W) + h * W + w];
            }
        }
    }
}

RgaPixelFormat toRgaPixelFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::NV12:
            return RgaPixelFormat::NV12;
        case PixelFormat::NV16:
            return RgaPixelFormat::NV16;
        case PixelFormat::RGB888:
            return RgaPixelFormat::RGB888;
        case PixelFormat::BGR888:
            return RgaPixelFormat::BGR888;
        default:
            return RgaPixelFormat::NV12;
    }
}

}  // namespace

LightTrackImpl::LightTrackImpl() = default;
LightTrackImpl::~LightTrackImpl() { shutdown(); }

void LightTrackImpl::setConfig(const LightTrackConfig& cfg) { cfg_ = cfg; }

bool LightTrackImpl::loadModels(const std::string& template_model,
                                 const std::string& search_model,
                                 const std::string& head_model) {
    // Auto scheduling: NPU çekirdek ataması RKNN SDK tarafından otomatik
    // yapılır. Referans LightTrack projesindeki (RKEngine) ile aynı davranış.
    // rknn_init(model, len, 0, NULL) — flag=0 → RKNN_NPU_CORE_AUTO (0).
    if (!template_engine_.initialize(template_model)) {
        std::cerr << "[LightTrackImpl] template_engine_ (init) yüklenemedi\n";
        return false;
    }
    if (!search_engine_.initialize(search_model)) {
        std::cerr << "[LightTrackImpl] search_engine_ (backbone) yüklenemedi\n";
        return false;
    }
    if (!head_engine_.initialize(head_model)) {
        std::cerr << "[LightTrackImpl] head_engine_ (neck_head) yüklenemedi\n";
        return false;
    }

    models_loaded_ = true;
    createWindow();
    createGrids();
    return true;
}

void LightTrackImpl::createWindow() {
    int sz = cfg_.score_size;
    std::vector<float> hanning(sz);
    for (int i = 0; i < sz; ++i) {
        hanning[i] = 0.5f - 0.5f * std::cos(2.0f * 3.1415926535898f * i / (sz - 1));
    }
    window_.assign(static_cast<size_t>(sz) * sz, 0.0f);
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            window_[i * sz + j] = hanning[i] * hanning[j];
        }
    }
}

void LightTrackImpl::createGrids() {
    int sz = cfg_.score_size;
    grid_x_.assign(static_cast<size_t>(sz) * sz, 0.0f);
    grid_y_.assign(static_cast<size_t>(sz) * sz, 0.0f);
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            grid_x_[i * sz + j] = static_cast<float>(j * cfg_.total_stride);
            grid_y_[i * sz + j] = static_cast<float>(i * cfg_.total_stride);
        }
    }
}

bool LightTrackImpl::initialize(const DmaBufferPtr& frame, const BBox& bbox) {
    if (!models_loaded_ || !frame) return false;

    state_.im_w = static_cast<int>(frame->width);
    state_.im_h = static_cast<int>(frame->height);

    state_.target_cx = bbox.x + (bbox.width - 1) / 2.0f;
    state_.target_cy = bbox.y + (bbox.height - 1) / 2.0f;
    state_.target_w = static_cast<float>(bbox.width);
    state_.target_h = static_cast<float>(bbox.height);

    float wc_z = state_.target_w + cfg_.context_amount * (state_.target_w + state_.target_h);
    float hc_z = state_.target_h + cfg_.context_amount * (state_.target_w + state_.target_h);
    float s_z = std::round(std::sqrt(wc_z * hc_z));

    RgaPixelFormat src_fmt = toRgaPixelFormat(frame->format);

    // ==================== TEMPLATE AUGMENTATION ====================
    // lighttrack.cpp init()'iyle AYNI mantık: birkaç hafif farklı kırpım
    // üretip template_engine_ çıktılarını ortalayarak tek kararlı zf
    // embedding'i elde ediyoruz. Bu implementasyonda 3 augmentasyon var
    // (base + yatay-flip + %10 büyük-ölçek) — parlaklık/karanlık
   // augmentasyonları RGA karşılığı doğrulanamadığı için ATLANDI
    // (bkz. LightTrackConfig::num_augments yorumu).
    //
    // aug_inputs: (ham piksel işaretçisi, boyut) çiftleri. Sadece
    // DmaBufferPtr TUTMUYORUZ çünkü flip augmentasyonu artık RGA
    // DEĞİL, CPU'da yapılıyor (aşağıdaki not) — o durumda arkasında bir
    // DmaBuffer yok, düz bir std::vector<uint8_t>.
    struct AugInput {
        const void* data;
        uint32_t size;
    };
    std::vector<AugInput> aug_inputs;
    std::vector<uint8_t> flipped_pixels;  // flip_crop'un ömrü boyunca canlı kalmalı

    DmaBufferPtr base_crop;
    if (!preprocessor_.cropTargetCentric(frame, src_fmt, state_.target_cx,
                                          state_.target_cy, static_cast<int>(s_z),
                                          cfg_.exemplar_size, state_.pad_r, state_.pad_g,
                                          state_.pad_b, base_crop)) {
        std::cerr << "[LightTrackImpl] initialize: temel crop başarısız\n";
        return false;
    }
    if (base_crop->virt_addr) {
        aug_inputs.push_back({base_crop->virt_addr, base_crop->size});
    }

    int n_augments = std::max(1, std::min(cfg_.num_augments, 3));

    // KRİTİK DÜZELTME (canlı RTSP testinde keşfedildi): RgaPreprocessor::
    // flipHorizontal (RGA imflip) KULLANILMIYOR — exemplar_size=127,
    // RGA'nın RGB888 kaynaklarda zorunlu kıldığı 16-byte genişlik stride
    // hizasına (127 % 16 != 0) uymuyor, "src unsupport width stride"
    // hatasıyla sessizce başarısız oluyordu (augmentasyon sadece
    // atlanıyordu, veri bozulması yoktu, ama template 3 yerine 2
    // augmentasyondan üretiliyordu). Bu işlem SADECE initialize()'da,
    // TEK SEFERLİK ve küçük (127×127×3) bir buffer üzerinde çalıştığından
    // (gerçek zamanlı FPS'e dokunmuyor — bkz. LightTrackImpl.hpp header
    // yorumu), RGA'ya hiç gerek yok: düz bir CPU flip yeterli ve stride
    // kısıtından tamamen bağımsız.
    if (n_augments > 1 && base_crop->virt_addr) {
        const int sz = cfg_.exemplar_size;
        const auto* src = static_cast<const uint8_t*>(base_crop->virt_addr);
        flipped_pixels.resize(static_cast<size_t>(sz) * sz * 3);
        for (int y = 0; y < sz; ++y) {
            const uint8_t* src_row = src + static_cast<size_t>(y) * sz * 3;
            uint8_t* dst_row = flipped_pixels.data() + static_cast<size_t>(y) * sz * 3;
            for (int x = 0; x < sz; ++x) {
                const uint8_t* s = src_row + static_cast<size_t>(sz - 1 - x) * 3;
                uint8_t* d = dst_row + static_cast<size_t>(x) * 3;
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            }
        }
        aug_inputs.push_back(
            {flipped_pixels.data(), static_cast<uint32_t>(flipped_pixels.size())});
    }

    DmaBufferPtr scaled_crop;  // ömrü aug_inputs kullanımı boyunca canlı kalmalı
    if (n_augments > 2) {
        if (preprocessor_.cropTargetCentric(frame, src_fmt, state_.target_cx,
                                             state_.target_cy,
                                             static_cast<int>(s_z * 1.1f),
                                             cfg_.exemplar_size, state_.pad_r,
                                             state_.pad_g, state_.pad_b, scaled_crop) &&
            scaled_crop->virt_addr) {
            aug_inputs.push_back({scaled_crop->virt_addr, scaled_crop->size});
        }
    }

    const auto& t_out_profiles = template_engine_.outputProfiles();
    if (t_out_profiles.empty()) {
        std::cerr << "[LightTrackImpl] template_engine_ çıktı profili boş\n";
        return false;
    }

    std::vector<float> accum(t_out_profiles[0].n_elems, 0.0f);
    int n_used = 0;

    for (const auto& in : aug_inputs) {
        // KRİTİK DÜZELTME (M5'te keşfedildi): bindInputZeroCopy DEĞİL —
        // bu modelin native girişi FLOAT16-container, RGA'nın ürettiği ham
        // UINT8 RGB888'i doğrudan zero-copy bağlamak "bayt yozlaşması"
        // riski taşıyor (ve pratikte "invalid tensor malloc size" ile
        // çöküyordu). RGA yine crop+resize'ı donanımda yaptı; burada
        // sadece küçük (127x127x3) bir CPU kopyası var, maliyeti
        // ihmal edilebilir.
        if (!template_engine_.setInputCopy(0, in.data, in.size,
                                            RknnInferenceEngine::InputDataType::UInt8Image)) {
            continue;
        }
        if (!template_engine_.run()) continue;

        RknnInferenceEngine::OutputView view;
        if (!template_engine_.getOutput(0, view)) {
            template_engine_.releaseOutputs();
            continue;
        }

        const float* out = static_cast<const float*>(view.data);
        for (uint32_t i = 0; i < t_out_profiles[0].n_elems; ++i) {
            accum[i] += out[i];
        }
        template_engine_.releaseOutputs();
        ++n_used;
    }

    if (n_used == 0) {
        std::cerr << "[LightTrackImpl] initialize: template_engine_ hiç başarılı "
                     "çalışmadı\n";
        return false;
    }

    std::vector<float> zf_avg(t_out_profiles[0].n_elems);
    for (uint32_t i = 0; i < t_out_profiles[0].n_elems; ++i) {
        zf_avg[i] = accum[i] / n_used;
    }

    int zC = static_cast<int>(t_out_profiles[0].dims[1]);
    int zH = static_cast<int>(t_out_profiles[0].dims[2]);
    int zW = static_cast<int>(t_out_profiles[0].dims[3]);
    zf_transposed_.assign(t_out_profiles[0].n_elems, 0.0f);
    transposeNchwToNhwc(zf_avg.data(), zf_transposed_.data(), zC, zH, zW);
    // ==================== TEMPLATE AUGMENTATION SONU ====================

    state_.last_good_w = state_.target_w;
    state_.last_good_h = state_.target_h;
    tracker_state_ = TrackerState::Tracking;

    std::cout << "[LightTrackImpl] initialize tamamlandı: " << n_used
              << "/" << aug_inputs.size() << " augmentasyon başarılı, zf boyutu="
              << zf_transposed_.size() << "\n";

    return true;
}

void LightTrackImpl::updateFromSearchCrop(const DmaBufferPtr& search_crop,
                                           float scale_z) {
    state_.cls_score_max = 0.0f;  // varsayılan: başarısızlık durumunda LOST'a düşer

    // Aynı düzeltme (bkz. initialize() içindeki not): zero-copy DEĞİL,
    // klasik kopya + doğru UInt8Image tip beyanı.
    if (!search_crop->virt_addr) return;

    uint64_t t0 = nowUs();
    if (!search_engine_.setInputCopy(0, search_crop->virt_addr, search_crop->size,
                                      RknnInferenceEngine::InputDataType::UInt8Image)) {
        return;
    }
    if (!search_engine_.run()) return;

    const auto& x_out_profiles = search_engine_.outputProfiles();
    if (x_out_profiles.empty()) return;

    RknnInferenceEngine::OutputView xf_view;
    if (!search_engine_.getOutput(0, xf_view)) {
        search_engine_.releaseOutputs();
        return;
    }

    int xC = static_cast<int>(x_out_profiles[0].dims[1]);
    int xH = static_cast<int>(x_out_profiles[0].dims[2]);
    int xW = static_cast<int>(x_out_profiles[0].dims[3]);
    xf_transposed_.assign(x_out_profiles[0].n_elems, 0.0f);
    transposeNchwToNhwc(static_cast<const float*>(xf_view.data), xf_transposed_.data(),
                         xC, xH, xW);
    search_engine_.releaseOutputs();

    uint64_t t1 = nowUs();
    uint64_t backbone_dt = t1 - t0;
    perf_backbone_us_ += backbone_dt;
    last_backbone_us_ = backbone_dt;
    int64_t npu_us = search_engine_.lastRunDurationUs();
    if (npu_us > 0) perf_backbone_npu_us_ += static_cast<uint64_t>(npu_us);

    if (zf_transposed_.empty()) return;  // initialize() hiç çalışmamış

    if (!head_engine_.setInputCopy(
            0, zf_transposed_.data(),
            static_cast<uint32_t>(zf_transposed_.size() * sizeof(float)),
            RknnInferenceEngine::InputDataType::Float32Tensor)) {
        return;
    }
    if (!head_engine_.setInputCopy(
            1, xf_transposed_.data(),
            static_cast<uint32_t>(xf_transposed_.size() * sizeof(float)),
            RknnInferenceEngine::InputDataType::Float32Tensor)) {
        return;
    }
    if (!head_engine_.run()) return;

    RknnInferenceEngine::OutputView cls_view, bbox_view;
    if (!head_engine_.getOutput(0, cls_view) || !head_engine_.getOutput(1, bbox_view)) {
        head_engine_.releaseOutputs();
        return;
    }

    uint64_t t2 = nowUs();
    uint64_t head_dt = t2 - t1;
    perf_head_us_ += head_dt;
    last_head_us_ = head_dt;
    int64_t head_npu_us = head_engine_.lastRunDurationUs();
    if (head_npu_us > 0) perf_head_npu_us_ += static_cast<uint64_t>(head_npu_us);

    int rows = static_cast<int>(cls_view.profile.dims[2]);
    int cols = static_cast<int>(cls_view.profile.dims[3]);
    int channels_cls = static_cast<int>(cls_view.profile.dims[1]);
    int channel_size = rows * cols;

    const float* cls_raw = static_cast<const float*>(cls_view.data);
    std::vector<float> cls_scores(channel_size);
    // channels_cls>=2: NanoTrack-tarzı bg/fg cikti uyumlulugu (lighttrack.cpp
    // ile ayni davranis). LightTrack native: tek kanal.
    if (channels_cls >= 2) {
        const float* target = cls_raw + channel_size;
        for (int i = 0; i < channel_size; ++i) cls_scores[i] = sigmoidFast(target[i]);
    } else {
        for (int i = 0; i < channel_size; ++i) cls_scores[i] = sigmoidFast(cls_raw[i]);
    }

    const float* bbox_raw = static_cast<const float*>(bbox_view.data);
    std::vector<float> pred_x1(channel_size), pred_y1(channel_size),
        pred_x2(channel_size), pred_y2(channel_size);
    std::vector<float> w(channel_size), h(channel_size);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            pred_x1[idx] = grid_x_[idx] - bbox_raw[idx + channel_size * 0];
            pred_y1[idx] = grid_y_[idx] - bbox_raw[idx + channel_size * 1];
            pred_x2[idx] = grid_x_[idx] + bbox_raw[idx + channel_size * 2];
            pred_y2[idx] = grid_y_[idx] + bbox_raw[idx + channel_size * 3];
            w[idx] = pred_x2[idx] - pred_x1[idx];
            h[idx] = pred_y2[idx] - pred_y1[idx];
        }
    }

    float scaled_target_w = state_.target_w * scale_z;
    float scaled_target_h = state_.target_h * scale_z;
    float sz_wh = szWhFun(scaled_target_w, scaled_target_h);
    float target_ratio = scaled_target_w / scaled_target_h;

    std::vector<float> penalty(channel_size);
    float max_score = 0.0f;
    int max_idx = 0;

    for (int i = 0; i < channel_size; ++i) {
        float pad = (w[i] + h[i]) * 0.5f;
        float s_c_raw = std::sqrt((w[i] + pad) * (h[i] + pad)) / sz_wh;
        float s_c = std::max(s_c_raw, 1.0f / s_c_raw);

        float r_c_raw = target_ratio / (w[i] / h[i]);
        float r_c = std::max(r_c_raw, 1.0f / r_c_raw);

        penalty[i] = std::exp(-1.0f * (s_c * r_c - 1.0f) * cfg_.penalty_k);
        float pscore = (penalty[i] * cls_scores[i]) * (1.0f - cfg_.window_influence) +
                       window_[i] * cfg_.window_influence;
        if (pscore > max_score) {
            max_score = pscore;
            max_idx = i;
        }
    }

    float pred_xs = (pred_x1[max_idx] + pred_x2[max_idx]) / 2.0f;
    float pred_ys = (pred_y1[max_idx] + pred_y2[max_idx]) / 2.0f;
    float pred_w = (pred_x2[max_idx] - pred_x1[max_idx]) / scale_z;
    float pred_h = (pred_y2[max_idx] - pred_y1[max_idx]) / scale_z;

    float diff_xs = (pred_xs - cfg_.instance_size / 2.0f) / scale_z;
    float diff_ys = (pred_ys - cfg_.instance_size / 2.0f) / scale_z;

    float lr = penalty[max_idx] * cls_scores[max_idx] * cfg_.lr;

    if (debug_frames_logged_ < 3) {
        float mn = 1e9f, mx = -1e9f, sum = 0.0f;
        for (int i = 0; i < channel_size; ++i) {
            mn = std::min(mn, cls_scores[i]);
            mx = std::max(mx, cls_scores[i]);
            sum += cls_scores[i];
        }
        std::cout << "[LightTrackImpl][debug " << debug_frames_logged_
                  << "] cls min=" << mn << " max=" << mx
                  << " ort=" << (sum / channel_size) << " peak=("
                  << (max_idx / cols) << "," << (max_idx % cols) << ")/"
                  << rows << "x" << cols << " diff=(" << diff_xs << ","
                  << diff_ys << ") lr=" << lr << "\n";
        ++debug_frames_logged_;
    }

    state_.target_cx = state_.target_cx + diff_xs;
    state_.target_cy = state_.target_cy + diff_ys;
    state_.target_w = pred_w * lr + (1.0f - lr) * state_.target_w;
    state_.target_h = pred_h * lr + (1.0f - lr) * state_.target_h;
    state_.cls_score_max = cls_scores[max_idx];

    head_engine_.releaseOutputs();

    uint64_t post_dt = nowUs() - t2;
    perf_post_us_ += post_dt;
    last_post_us_ = post_dt;
}

TrackResult LightTrackImpl::track(const DmaBufferPtr& frame) {
    TrackResult result;
    if (tracker_state_ == TrackerState::Idle || !frame) {
        return result;
    }

    float hc_z = state_.target_h + cfg_.context_amount * (state_.target_w + state_.target_h);
    float wc_z = state_.target_w + cfg_.context_amount * (state_.target_w + state_.target_h);
    float s_z = std::sqrt(wc_z * hc_z);
    float scale_z = cfg_.exemplar_size / s_z;

    float d_search = (cfg_.instance_size - cfg_.exemplar_size) / 2.0f;
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;

    uint64_t crop_t0 = nowUs();
    DmaBufferPtr search_crop;
    if (!preprocessor_.cropTargetCentric(frame, toRgaPixelFormat(frame->format),
                                          state_.target_cx, state_.target_cy,
                                          static_cast<int>(s_x), cfg_.instance_size,
                                          state_.pad_r, state_.pad_g, state_.pad_b,
                                          search_crop)) {
        return result;
    }
    uint64_t crop_dt = nowUs() - crop_t0;
    perf_crop_us_ += crop_dt;
    last_crop_us_ = crop_dt;

    updateFromSearchCrop(search_crop, scale_z);

    if (++perf_frames_ >= 60) {
        std::cout << "[LightTrackImpl][perf] son " << perf_frames_
                  << " kare ortalaması: crop=" << (perf_crop_us_ / perf_frames_)
                  << "us backbone=" << (perf_backbone_us_ / perf_frames_)
                  << "us (npu=" << (perf_backbone_npu_us_ / perf_frames_)
                  << "us) head=" << (perf_head_us_ / perf_frames_)
                  << "us (npu=" << (perf_head_npu_us_ / perf_frames_)
                  << "us) post=" << (perf_post_us_ / perf_frames_) << "us\n";
        perf_crop_us_ = perf_backbone_us_ = perf_head_us_ = perf_post_us_ = 0;
        perf_backbone_npu_us_ = perf_head_npu_us_ = 0;
        perf_frames_ = 0;
    }

    state_.target_cx =
        std::max(0.0f, std::min(static_cast<float>(state_.im_w), state_.target_cx));
    state_.target_cy =
        std::max(0.0f, std::min(static_cast<float>(state_.im_h), state_.target_cy));
    state_.target_w =
        std::max(10.0f, std::min(static_cast<float>(state_.im_w), state_.target_w));
    state_.target_h =
        std::max(10.0f, std::min(static_cast<float>(state_.im_h), state_.target_h));

    // ---- KAYIP HEDEF GÜVENLİK AĞI (lighttrack.cpp ile aynı) ----
    if (state_.cls_score_max < cfg_.lost_score_threshold) {
        state_.target_w = state_.last_good_w;
        state_.target_h = state_.last_good_h;
        tracker_state_ = TrackerState::Lost;
    } else {
        state_.last_good_w = state_.target_w;
        state_.last_good_h = state_.target_h;
        tracker_state_ = TrackerState::Tracking;
    }

    result.bbox.x = static_cast<int>(state_.target_cx - state_.target_w / 2);
    result.bbox.y = static_cast<int>(state_.target_cy - state_.target_h / 2);
    result.bbox.width = static_cast<int>(state_.target_w);
    result.bbox.height = static_cast<int>(state_.target_h);
    result.confidence = state_.cls_score_max;
    return result;
}

TrackerState LightTrackImpl::getState() const { return tracker_state_; }

TrackTiming LightTrackImpl::lastTiming() const {
    TrackTiming t;
    t.crop_us = last_crop_us_;
    t.invoke_us = last_backbone_us_ + last_head_us_;
    t.postproc_us = last_post_us_;
    return t;
}

void LightTrackImpl::shutdown() {
    template_engine_.shutdown();
    search_engine_.shutdown();
    head_engine_.shutdown();
    models_loaded_ = false;
    tracker_state_ = TrackerState::Idle;
}