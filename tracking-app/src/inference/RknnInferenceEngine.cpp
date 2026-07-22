#include "inference/RknnInferenceEngine.hpp"

#include "rknn_api.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

Precision detectPrecision(const rknn_tensor_attr& attr) {
    if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
        return Precision::INT8;
    }
    if (attr.type == RKNN_TENSOR_FLOAT16 && attr.qnt_type == RKNN_TENSOR_QNT_NONE) {
        return Precision::FP16;
    }
    return Precision::UNKNOWN;
}

TensorProfile toProfile(const rknn_tensor_attr& attr) {
    TensorProfile p;
    p.index = attr.index;
    p.name = attr.name;
    p.n_dims = attr.n_dims;
    for (uint32_t i = 0; i < attr.n_dims && i < 4; ++i) p.dims[i] = attr.dims[i];
    p.n_elems = attr.n_elems;
    p.size = attr.size;
    p.w_stride = attr.w_stride;
    p.h_stride = attr.h_stride;
    p.size_with_stride = attr.size_with_stride;
    p.precision = detectPrecision(attr);
    p.zp = attr.zp;
    p.scale = attr.scale;
    p.raw_type = static_cast<int32_t>(attr.type);
    p.raw_fmt = static_cast<int32_t>(attr.fmt);
    return p;
}

std::vector<uint8_t> readModelFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) return {};
    return buf;
}

const char* precisionName(Precision p) {
    switch (p) {
        case Precision::FP16:
            return "FP16";
        case Precision::INT8:
            return "INT8";
        default:
            return "?";
    }
}

}  // namespace

struct RknnInferenceEngine::Impl {
    rknn_context ctx = 0;
    bool ctx_created = false;

    std::vector<TensorProfile> input_profiles;
    std::vector<TensorProfile> output_profiles;

    // Zero-copy ile bağlanan girdi mem handle'ları — shutdown'da
    // rknn_destroy_mem ile serbest bırakılmaları gerekiyor.
    std::vector<rknn_tensor_mem*> bound_input_mems;

    // KRİTİK DÜZELTME (canlı M6 testinde keşfedildi — "donmuş bbox"un kök
    // nedeni): rknn_inputs_set her tensör için AYRI AYRI (n=1) çağrılamaz.
    // Referans uygulama (rknn_engine.cpp) TÜM girişleri tek listede,
    // TEK rknn_inputs_set(ctx, n, array) çağrısıyla veriyor. Bizim eski
    // setInputCopy'miz her tensörü ayrı çağrıyla set ediyordu — tek girişli
    // modellerde (template/search) fark yok, ama 2 girişli head modelinde
    // ikinci çağrı ilk girişin iç ön-işleme durumunu bozuyor: RKNN_SUCC
    // döner, çökmez, ama head çıktısı çöp olur -> sigmoid skorları düz
    // ~0.5 -> pscore tamamen Hanning penceresine yaslanır -> tepe hep
    // arama merkezinde -> bbox HİÇ hareket etmez. M5 bunu yakalayamadı
    // çünkü sadece "çöküyor mu"ya baktı, çıktı doğruluğunu ölçmedi.
    // Çözüm: setInputCopy girişleri burada biriktirir, run() hepsini
    // tek rknn_inputs_set çağrısıyla verir (referansla aynı semantik).
    std::vector<rknn_input> pending_inputs;

    std::vector<rknn_output> pending_outputs;
    bool outputs_fetched = false;

    ~Impl() { shutdown(); }

    void shutdown() {
        for (auto* mem : bound_input_mems) {
            if (mem) rknn_destroy_mem(ctx, mem);
        }
        bound_input_mems.clear();

        releaseOutputsInternal();

        if (ctx_created) {
            rknn_destroy(ctx);
            ctx_created = false;
        }
    }

    void releaseOutputsInternal() {
        if (outputs_fetched && !pending_outputs.empty()) {
            rknn_outputs_release(ctx, static_cast<uint32_t>(pending_outputs.size()),
                                  pending_outputs.data());
        }
        pending_outputs.clear();
        outputs_fetched = false;
    }
};

RknnInferenceEngine::RknnInferenceEngine() : impl_(std::make_unique<Impl>()) {}
RknnInferenceEngine::~RknnInferenceEngine() = default;

bool RknnInferenceEngine::initialize(const std::string& model_path) {
    auto model_data = readModelFile(model_path);
    if (model_data.empty()) {
        std::cerr << "[RknnInferenceEngine] Model dosyası okunamadı: " << model_path
                  << "\n";
        return false;
    }

    // KRİTİK DÜZELTME (M5'te 3 farklı hipotez denendikten sonra keşfedildi):
    // RKNN_FLAG_MEM_ALLOC_OUTSIDE burada KULLANILMIYOR artık. Bu flag,
    // driver'a "tüm bellek dışarıdan rknn_set_io_mem ile verilecek" der —
    // ve klasik rknn_inputs_set yoluyla (bizim template/search/head
    // engine'lerinin hepsinin kullandığı yol) UYUŞMUYOR. Bu flag açıkken
    // rknn_inputs_set çağrısı, verdiğimiz input.type/size ne olursa olsun
    // "invalid tensor malloc size... size: 0" ile çöküyordu — referans
    // kod (rknn_engine.cpp) da zaten flag=0 kullanıyor. Zero-copy'ye
    // (bindInputZeroCopy) artık şu an hiçbir yerde ihtiyaç duyulmuyor
    // (bu modellerin native formatı — FLOAT16 container — zaten zero-copy
    // ile uyumsuzdu, ayrı bir keşifle bulmuştuk). Gelecekte zero-copy
    // gerçekten gerekiyorsa (örn. uyumlu bir detector modeliyle), o motor
    // için AYRI bir initialize() yolu/flag'i düşünülmeli — aynı context
    // ikisini birden karışık kullanamıyor.
    int ret = rknn_init(&impl_->ctx, model_data.data(),
                         static_cast<uint32_t>(model_data.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[RknnInferenceEngine] rknn_init başarısız (" << model_path
                  << "): ret=" << ret << "\n";
        return false;
    }
    impl_->ctx_created = true;

    // Auto scheduling: rknn_set_core_mask ÇAĞRILMIYOR.
    // RKNN_NPU_CORE_AUTO (0) varsayılan — RKNN SDK çekirdek atamasını
    // kendisi yapar. Bu, referans LightTrack projesindeki ile aynı davranış.

    rknn_input_output_num io_num{};
    ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        std::cerr << "[RknnInferenceEngine] IN_OUT_NUM sorgusu başarısız\n";
        return false;
    }

    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "[RknnInferenceEngine] INPUT_ATTR[" << i
                      << "] sorgusu başarısız\n";
            return false;
        }
        impl_->input_profiles.push_back(toProfile(attr));
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "[RknnInferenceEngine] OUTPUT_ATTR[" << i
                      << "] sorgusu başarısız\n";
            return false;
        }
        impl_->output_profiles.push_back(toProfile(attr));
    }

    impl_->bound_input_mems.assign(io_num.n_input, nullptr);

    std::cout << "[RknnInferenceEngine] " << model_path << " yüklendi: "
              << io_num.n_input << " girdi, " << io_num.n_output << " çıktı\n";
    for (auto& p : impl_->input_profiles) {
        std::cout << "  girdi[" << p.index << "] " << p.name << " dims=("
                  << p.dims[0] << "," << p.dims[1] << "," << p.dims[2] << ","
                  << p.dims[3] << ") precision=" << precisionName(p.precision)
                  << " raw_type=" << p.raw_type << " raw_fmt=" << p.raw_fmt
                  << " n_elems=" << p.n_elems << " size=" << p.size
                  << " w_stride=" << p.w_stride
                  << " size_with_stride=" << p.size_with_stride << "\n";
    }
    for (auto& p : impl_->output_profiles) {
        std::cout << "  çıktı[" << p.index << "] " << p.name << " dims=("
                  << p.dims[0] << "," << p.dims[1] << "," << p.dims[2] << ","
                  << p.dims[3] << ") precision=" << precisionName(p.precision)
                  << "\n";
    }

    return true;
}

const std::vector<TensorProfile>& RknnInferenceEngine::inputProfiles() const {
    return impl_->input_profiles;
}

const std::vector<TensorProfile>& RknnInferenceEngine::outputProfiles() const {
    return impl_->output_profiles;
}

bool RknnInferenceEngine::bindInputZeroCopy(uint32_t tensor_index,
                                             const DmaBufferPtr& buffer) {
    if (!buffer || buffer->fd < 0 || tensor_index >= impl_->input_profiles.size()) {
        return false;
    }

    if (impl_->bound_input_mems[tensor_index]) {
        rknn_destroy_mem(impl_->ctx, impl_->bound_input_mems[tensor_index]);
        impl_->bound_input_mems[tensor_index] = nullptr;
    }

    rknn_tensor_mem* mem = rknn_create_mem_from_fd(impl_->ctx, buffer->fd,
                                                    buffer->virt_addr, buffer->size, 0);
    if (!mem) {
        std::cerr << "[RknnInferenceEngine] rknn_create_mem_from_fd başarısız "
                     "(tensor_index="
                  << tensor_index << ")\n";
        return false;
    }

    rknn_tensor_attr attr{};
    attr.index = tensor_index;
    int ret = rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN_SUCC) {
        rknn_destroy_mem(impl_->ctx, mem);
        return false;
    }

    ret = rknn_set_io_mem(impl_->ctx, mem, &attr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[RknnInferenceEngine] rknn_set_io_mem başarısız (zero-copy)\n";
        rknn_destroy_mem(impl_->ctx, mem);
        return false;
    }

    impl_->bound_input_mems[tensor_index] = mem;
    return true;
}

bool RknnInferenceEngine::setInputCopy(uint32_t tensor_index, const void* data,
                                        uint32_t size, InputDataType data_type) {
    if (!data || tensor_index >= impl_->input_profiles.size()) {
        return false;
    }

    // KRİTİK: input.type BİZİM verdiğimiz verinin gerçek tipini belirtir,
    // modelin sorgulanan (native) tipini DEĞİL. Referans engine_helper.h
    // (nn_tensor_attr_to_cvimg_input_data) de aynı ilkeyi uyguluyor —
    // kamera verisi her zaman UINT8/NHWC olarak bildirilir, model native
    // olarak FLOAT16 bekliyor olsa bile (RKNN pass_through=0 iken kendi
    // içinde dönüştürür).
    rknn_input input{};
    input.index = tensor_index;
    input.buf = const_cast<void*>(data);
    input.size = size;
    input.pass_through = 0;

    if (data_type == InputDataType::UInt8Image) {
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
    } else {  // Float32Tensor
        input.type = RKNN_TENSOR_FLOAT32;
        input.fmt =
            static_cast<rknn_tensor_format>(impl_->input_profiles[tensor_index].raw_fmt);
    }

    // rknn_inputs_set BURADA ÇAĞRILMAZ — girişler biriktirilir, run()
    // hepsini TEK çağrıyla verir (bkz. Impl::pending_inputs yorumu).
    // Bu yüzden `data` işaretçisi run() dönene kadar canlı kalmak ZORUNDA
    // (mevcut tüm çağıranlar bunu zaten sağlıyor: zf/xf üye vektörler,
    // crop DmaBufferPtr'ları run()'u kapsayan scope'larda).
    for (auto& pi : impl_->pending_inputs) {
        if (pi.index == tensor_index) {
            pi = input;
            return true;
        }
    }
    impl_->pending_inputs.push_back(input);
    return true;
}

bool RknnInferenceEngine::run() {
    impl_->releaseOutputsInternal();

    if (!impl_->pending_inputs.empty()) {
        std::sort(impl_->pending_inputs.begin(), impl_->pending_inputs.end(),
                  [](const rknn_input& a, const rknn_input& b) { return a.index < b.index; });
        int ret = rknn_inputs_set(impl_->ctx,
                                   static_cast<uint32_t>(impl_->pending_inputs.size()),
                                   impl_->pending_inputs.data());
        impl_->pending_inputs.clear();
        if (ret != RKNN_SUCC) {
            std::cerr << "[RknnInferenceEngine] rknn_inputs_set başarısız: ret=" << ret
                      << "\n";
            return false;
        }
    }

    int ret = rknn_run(impl_->ctx, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[RknnInferenceEngine] rknn_run başarısız: ret=" << ret << "\n";
        return false;
    }
    return true;
}

bool RknnInferenceEngine::getOutput(uint32_t tensor_index, OutputView& out) {
    if (tensor_index >= impl_->output_profiles.size()) {
        return false;
    }

    if (!impl_->outputs_fetched) {
        size_t n = impl_->output_profiles.size();
        impl_->pending_outputs.assign(n, rknn_output{});
        for (size_t i = 0; i < n; ++i) {
            // want_float HER ZAMAN zorlanır — bkz. header notu.
            impl_->pending_outputs[i].want_float = 1;
            impl_->pending_outputs[i].index = static_cast<uint32_t>(i);
        }
        int ret = rknn_outputs_get(impl_->ctx, static_cast<uint32_t>(n),
                                    impl_->pending_outputs.data(), nullptr);
        if (ret != RKNN_SUCC) {
            std::cerr << "[RknnInferenceEngine] rknn_outputs_get başarısız: ret="
                      << ret << "\n";
            return false;
        }
        impl_->outputs_fetched = true;
    }

    const auto& ro = impl_->pending_outputs[tensor_index];
    out.data = ro.buf;
    out.size = ro.size;
    out.profile = impl_->output_profiles[tensor_index];
    return true;
}

void RknnInferenceEngine::releaseOutputs() { impl_->releaseOutputsInternal(); }

int64_t RknnInferenceEngine::lastRunDurationUs() const {
    rknn_perf_run perf{};
    int ret = rknn_query(impl_->ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf));
    if (ret != RKNN_SUCC) {
        return -1;
    }
    return perf.run_duration;
}

void RknnInferenceEngine::shutdown() { impl_->shutdown(); }