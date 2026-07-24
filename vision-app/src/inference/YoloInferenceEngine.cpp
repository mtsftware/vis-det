#include "inference/YoloInferenceEngine.hpp"

#include "rknn_api.h"

#include <algorithm>
#include <chrono>
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
    TensorInfo output_info_;

    // Giris biriktirici — run()'da tek rknn_inputs_set ile verilir.
    rknn_input pending_input_;

    // Cikti — run() sonra rknn_outputs_get ile alinir.
    bool outputs_fetched = false;
    std::vector<rknn_output> pending_outputs;

    ~Impl() { unload(); }

    void unload() {
        if (ctx_created && ctx) {
            // Cikti serbest birak
            if (outputs_fetched && !pending_outputs.empty()) {
                rknn_outputs_release(ctx, static_cast<uint32_t>(pending_outputs.size()),
                                     pending_outputs.data());
            }
            // NPU context yok et
            rknn_destroy(ctx);
            ctx = 0;
            ctx_created = false;
        }
        outputs_fetched = false;
        pending_outputs.clear();
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

    // Input profil sorgula
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
            impl_->input_info_.zp = attr.zp;
            impl_->input_info_.scale = attr.scale;
            std::cout << "[YoloInferenceEngine] Girdi: " << attr.name
                      << " dims=(" << impl_->input_info_.dims[0] << ","
                      << impl_->input_info_.dims[1] << "," << impl_->input_info_.dims[2] << ","
                      << impl_->input_info_.dims[3] << ") type=" << attr.type
                      << " qnt_type=" << attr.qnt_type
                      << " zp=" << attr.zp << " scale=" << attr.scale << "\n";
        }
    }

    // Output tensor bilgisi
    if (io_num.n_output > 0) {
        rknn_tensor_attr attr{};
        attr.index = 0;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret == RKNN_SUCC) {
            impl_->output_info_.name = attr.name;
            impl_->output_info_.n_dims = attr.n_dims;
            for (uint32_t i = 0; i < attr.n_dims && i < 4; ++i) {
                impl_->output_info_.dims[i] = attr.dims[i];
            }
            impl_->output_info_.size = attr.size;
            impl_->output_info_.zp = attr.zp;
            impl_->output_info_.scale = attr.scale;
            std::cout << "[YoloInferenceEngine] Cikti: " << attr.name
                      << " dims=(" << impl_->output_info_.dims[0] << ","
                      << impl_->output_info_.dims[1] << "," << impl_->output_info_.dims[2] << ")"
                      << " size=" << attr.size << " zp=" << attr.zp
                      << " scale=" << attr.scale << "\n";
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

    // NWP calistir — süre ölçümü
    auto run_start = std::chrono::steady_clock::now();
    ret = rknn_run(impl_->ctx, nullptr);
    auto run_end = std::chrono::steady_clock::now();
    auto run_ms = std::chrono::duration_cast<std::chrono::milliseconds>(run_end - run_start).count();
    
    if (ret != RKNN_SUCC) {
        std::cerr << "[YoloInferenceEngine] rknn_run basarisiz, ret=" << ret << "\n";
        return false;
    }
    std::cout << "[YoloInferenceEngine] RKNN Run suresi: " << run_ms << " ms\n";
    return true;
}

bool YoloInferenceEngine::getOutput(const void*& output_data, uint32_t& output_size) {
    // Ciktilari al — want_float=1 ise runtime dequant yapar (float32)
    rknn_tensor_attr out_attr{};
    out_attr.index = 0;
    rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));

    // Boyutlari kontrol et
    uint32_t expected_size = static_cast<uint32_t>(out_attr.n_elems) * 4;  // float32

    // Once onceki ciktilari bosalt
    if (impl_->outputs_fetched && !impl_->pending_outputs.empty()) {
        rknn_outputs_release(impl_->ctx, static_cast<uint32_t>(impl_->pending_outputs.size()),
                             impl_->pending_outputs.data());
    }

    impl_->pending_outputs.assign(1, rknn_output{});
    impl_->pending_outputs[0].want_float = 1;  // auto dequant
    impl_->pending_outputs[0].index = 0;

    int ret = rknn_outputs_get(impl_->ctx, 1, impl_->pending_outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[YoloInferenceEngine] rknn_outputs_get basarisiz, ret=" << ret << "\n";
        return false;
    }
    impl_->outputs_fetched = true;

    const auto& out = impl_->pending_outputs[0];
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

const YoloInferenceEngine::TensorInfo& YoloInferenceEngine::outputInfo() const {
    return impl_->output_info_;
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