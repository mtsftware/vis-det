#include "inference/YoloInferenceEngine.hpp"

#include "rknn_api.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

std::vector<uint8_t> readModelFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) return {};
    return buf;
}

}  // namespace

struct YoloInferenceEngine::Impl {
    rknn_context ctx = 0;
    bool ctx_created = false;

    TensorInfo input_info_;
    std::vector<TensorInfo> output_infos_;  // io_num.n_output eleman (1 ya da 9)
    TensorInfo invalid_output_info_;        // outputInfo(index) sinir-disi donusu

    // Giris biriktirici — run()'da tek rknn_inputs_set ile verilir.
    rknn_input pending_input_;

    // Cikti — fetchOutputs() TEK rknn_outputs_get cagrisiyla TUM tensorleri
    // alir (RKNN cok-cikti modellerde boyle bekler, tek tek degil).
    bool outputs_fetched = false;
    std::vector<rknn_output> pending_outputs;

    ~Impl() { unload(); }

    void unload() {
        if (ctx_created && ctx) {
            if (outputs_fetched && !pending_outputs.empty()) {
                rknn_outputs_release(ctx, static_cast<uint32_t>(pending_outputs.size()),
                                     pending_outputs.data());
            }
            rknn_destroy(ctx);
            ctx = 0;
            ctx_created = false;
        }
        outputs_fetched = false;
        pending_outputs.clear();
        output_infos_.clear();
        std::memset(&pending_input_, 0, sizeof(pending_input_));
    }
};

YoloInferenceEngine::YoloInferenceEngine() : impl_(std::make_unique<Impl>()) {}
YoloInferenceEngine::~YoloInferenceEngine() = default;

bool YoloInferenceEngine::loadModel(const std::string& rknn_model_path) {
    auto model_data = readModelFile(rknn_model_path);
    if (model_data.empty()) {
        std::cerr << "[YoloInferenceEngine] Model dosyasi okunamadik: " << rknn_model_path << "\n";
        return false;
    }

    // RKNN context yukle — flag=0 (auto scheduling NPU)
    int ret = rknn_init(&impl_->ctx, model_data.data(),
                        static_cast<uint32_t>(model_data.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[YoloInferenceEngine] rknn_init basarisiz, ret=" << ret << "\n";
        return false;
    }
    impl_->ctx_created = true;

    // Input/Output sayisini sorgula
    rknn_input_output_num io_num{};
    ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        std::cerr << "[YoloInferenceEngine] IN_OUT_NUM sorgusu basarisiz\n";
        return false;
    }

    // Input tensor bilgisi
    if (io_num.n_input > 0) {
        rknn_tensor_attr attr{};
        attr.index = 0;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        if (ret == RKNN_SUCC) {
            impl_->input_info_.name = attr.name;
            impl_->input_info_.n_dims = attr.n_dims;
            for (uint32_t i = 0; i < attr.n_dims && i < 4; ++i) {
                impl_->input_info_.dims[i] = attr.dims[i];
            }
            impl_->input_info_.size = attr.size;
            impl_->input_info_.n_elems = attr.n_elems;
            impl_->input_info_.zp = attr.zp;
            impl_->input_info_.scale = attr.scale;
            impl_->input_info_.fmt = static_cast<int>(attr.fmt);
            std::cout << "[YoloInferenceEngine] Girdi: " << attr.name
                      << " dims=(" << impl_->input_info_.dims[0] << ","
                      << impl_->input_info_.dims[1] << "," << impl_->input_info_.dims[2] << ","
                      << impl_->input_info_.dims[3] << ") type=" << attr.type
                      << " qnt_type=" << attr.qnt_type
                      << " zp=" << attr.zp << " scale=" << attr.scale << "\n";
        }
    }

    // TUM cikti tensorlerinin bilgisini sorgula (1 ya da 9 — modele gore).
    // TENSOR LAYOUT UYARISI (edge-ai-workshop-rknn/inference.py'deki ayni
    // notun C++ karsiligi): DFL decode'un dogru calismasi icin box/score
    // tensorleri NCHW (fmt=0) olmali. NHWC (fmt=1) raporlanirsa postprocess
    // YANLIS SONUC uretir — asagida uyari basilir.
    impl_->output_infos_.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "[YoloInferenceEngine] OUTPUT_ATTR[" << i << "] sorgusu basarisiz\n";
            continue;
        }
        TensorInfo& info = impl_->output_infos_[i];
        info.name = attr.name;
        info.n_dims = attr.n_dims;
        for (uint32_t d = 0; d < attr.n_dims && d < 4; ++d) {
            info.dims[d] = attr.dims[d];
        }
        info.size = attr.size;
        info.n_elems = attr.n_elems;
        info.zp = attr.zp;
        info.scale = attr.scale;
        info.fmt = static_cast<int>(attr.fmt);

        std::cout << "[YoloInferenceEngine] Cikti[" << i << "]: " << attr.name
                  << " dims=(" << info.dims[0] << "," << info.dims[1] << ","
                  << info.dims[2] << "," << info.dims[3] << ")"
                  << " fmt=" << info.fmt << " (0=NCHW,1=NHWC)"
                  << " size=" << attr.size << " zp=" << attr.zp
                  << " scale=" << attr.scale << "\n";
        if (info.n_dims >= 3 && info.fmt != 0 /* RKNN_TENSOR_NCHW */) {
            std::cerr << "[YoloInferenceEngine] UYARI: cikti[" << i << "] fmt=NHWC "
                         "bekleniyordu NCHW — DFL decode YANLIS sonuc uretebilir "
                         "(YoloPostProcessor NCHW varsayiyor)\n";
        }
    }

    std::cout << "[YoloInferenceEngine] Model yuklendi: " << rknn_model_path
              << " (" << io_num.n_input << " girdi, " << io_num.n_output << " cikti)\n";
    return true;
}

bool YoloInferenceEngine::run(const void* input_buf, uint32_t input_size) {
    // Giris hazirla
    impl_->pending_input_.index = 0;
    impl_->pending_input_.buf = const_cast<void*>(input_buf);
    impl_->pending_input_.size = input_size;
    impl_->pending_input_.pass_through = 0;  // RKNN otomatik INT8 quantize edecek
    impl_->pending_input_.type = RKNN_TENSOR_UINT8;
    impl_->pending_input_.fmt = RKNN_TENSOR_NHWC;

    // Tek rknn_inputs_set cagri
    int ret = rknn_inputs_set(impl_->ctx, 1, &impl_->pending_input_);
    if (ret != RKNN_SUCC) {
        std::cerr << "[YoloInferenceEngine] rknn_inputs_set basarisiz, ret=" << ret << "\n";
        return false;
    }

    // NPU calistir. Sure olcumu artik burada HER KAREDE loglanmiyor (terminali
    // bogan gurultuydu) — PipelineOrchestrator, run()+fetchOutputs()'u saran
    // kendi wall-clock olcumunu periyodik ozet satirinda basiyor (bkz.
    // PipelineOrchestrator::Stats::last_inference_ms).
    ret = rknn_run(impl_->ctx, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[YoloInferenceEngine] rknn_run basarisiz, ret=" << ret << "\n";
        return false;
    }
    return true;
}

bool YoloInferenceEngine::fetchOutputs() {
    const uint32_t n = static_cast<uint32_t>(impl_->output_infos_.size());
    if (n == 0) {
        std::cerr << "[YoloInferenceEngine] fetchOutputs: cikti tensoru yok (model yuklendi mi?)\n";
        return false;
    }

    // Onceki ciktilari bosalt
    if (impl_->outputs_fetched && !impl_->pending_outputs.empty()) {
        rknn_outputs_release(impl_->ctx, static_cast<uint32_t>(impl_->pending_outputs.size()),
                             impl_->pending_outputs.data());
    }

    impl_->pending_outputs.assign(n, rknn_output{});
    for (uint32_t i = 0; i < n; ++i) {
        impl_->pending_outputs[i].want_float = 1;  // auto dequant
        impl_->pending_outputs[i].index = i;
    }

    // TUM ciktiler TEK cagriyla — RKNN cok-cikti modellerde bunu bekler.
    int ret = rknn_outputs_get(impl_->ctx, n, impl_->pending_outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[YoloInferenceEngine] rknn_outputs_get basarisiz, ret=" << ret << "\n";
        impl_->outputs_fetched = false;
        return false;
    }
    impl_->outputs_fetched = true;
    return true;
}

int YoloInferenceEngine::outputCount() const {
    return static_cast<int>(impl_->output_infos_.size());
}

bool YoloInferenceEngine::getOutput(int index, const void*& output_data,
                                     uint32_t& output_size) const {
    if (!impl_->outputs_fetched) {
        std::cerr << "[YoloInferenceEngine] getOutput: fetchOutputs() once cagirilmadi\n";
        return false;
    }
    if (index < 0 || static_cast<size_t>(index) >= impl_->pending_outputs.size()) {
        std::cerr << "[YoloInferenceEngine] getOutput: gecersiz index " << index << "\n";
        return false;
    }
    const auto& out = impl_->pending_outputs[static_cast<size_t>(index)];
    output_data = out.buf;
    output_size = out.size;
    return true;
}

void YoloInferenceEngine::releaseOutputs() {
    if (impl_->outputs_fetched && !impl_->pending_outputs.empty()) {
        rknn_outputs_release(impl_->ctx, static_cast<uint32_t>(impl_->pending_outputs.size()),
                             impl_->pending_outputs.data());
        impl_->pending_outputs.clear();
        impl_->outputs_fetched = false;
    }
}

const YoloInferenceEngine::TensorInfo& YoloInferenceEngine::inputInfo() const {
    return impl_->input_info_;
}

const YoloInferenceEngine::TensorInfo& YoloInferenceEngine::outputInfo(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= impl_->output_infos_.size()) {
        return impl_->invalid_output_info_;
    }
    return impl_->output_infos_[static_cast<size_t>(index)];
}

int64_t YoloInferenceEngine::lastRunDurationUs() const {
    rknn_perf_run perf{};
    int ret = rknn_query(impl_->ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf));
    if (ret != RKNN_SUCC) return -1;
    return perf.run_duration;
}

void YoloInferenceEngine::unload() {
    if (impl_) impl_->unload();
}
