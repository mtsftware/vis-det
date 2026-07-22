// M5 doğrulama: 3 LightTrack modelini otomatik çekirdek atamasıyla (AUTO)
// yükler, klasik giriş bağlama (setInputCopy) ile GERÇEKTEN çalıştırır.
//
// KRİTİK DÜZELTME (bu sürümde): input.type ARTIK modelin sorgulanan
// (native) tipinden DEĞİL, bizim verdiğimiz verinin GERÇEK tipinden
// belirleniyor — referans kodun (engine_helper.h) yaptığı gibi. Modelin
// attr.type=FLOAT16 raporlaması, girişin native/iç temsili ile ilgili;
// biz ham UINT8 piksel veriyoruz, RKNN pass_through=0 ile kendi içinde
// dönüştürüyor. Bu ayrımı gözden kaçırmak "invalid tensor malloc size...
// size: 0" çökmesine yol açıyordu.
//
// Bu yüzden ZERO-COPY (bindInputZeroCopy) template_engine_/search_engine_
// için ARTIK KULLANILMIYOR — bu modellerin native formatı (FLOAT16-
// container) RGA'nın üretebileceği ham UINT8 pikselle uyuşmuyor. RGA yine
// kırpma/ölçekleme/format işini donanımda yapacak, ama son adım (127x127x3
// ya da 288x288x3 gibi küçük veri) klasik bir CPU kopyası olacak — bu
// boyutlarda maliyeti ihmal edilebilir (M5'in bir önceki turunda
// tartışıldığı gibi).
//
// Derleme:
//   g++ -std=c++14 -O2 \
//     src/inference/RknnInferenceEngine.cpp \
//     src/inference/main_m5_test.cpp \
//     -Iinclude -Ilibrknn_api/include \
//     -Llibrknn_api/aarch64 -lrknnrt \
//     -Wl,-rpath,librknn_api/aarch64 \
//     -lpthread -o m5_engine_test
//
// Çalıştırma:
//   ./m5_engine_test models/lighttrack_init_fp16.rknn \
//                     models/lighttrack_backbone_fp16.rknn \
//                     models/lighttrack_neck_head_fp16.rknn

#include "inference/RknnInferenceEngine.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

// template_engine_ / search_engine_: ham UINT8 piksel verisi (RGA
// çıktısının gerçek karşılığı — burada dummy sıfır dolu).
bool runImageInput(const std::string& name, RknnInferenceEngine& engine) {
    const auto& inputs = engine.inputProfiles();
    if (inputs.empty()) {
        std::cerr << "[M5] " << name << " girdi profili boş\n";
        return false;
    }

    // n_elems zaten dims'ten (1*H*W*3) hesaplanmış, UINT8 için 1 byte/eleman.
    uint32_t sz = inputs[0].n_elems;
    std::vector<uint8_t> dummy(sz, 0);

    if (!engine.setInputCopy(0, dummy.data(), static_cast<uint32_t>(dummy.size()),
                              RknnInferenceEngine::InputDataType::UInt8Image)) {
        std::cerr << "[M5] " << name << " girdi bağlanamadı\n";
        return false;
    }

    if (!engine.run()) {
        std::cerr << "[M5] " << name << " run() başarısız\n";
        return false;
    }

    RknnInferenceEngine::OutputView view;
    if (!engine.getOutput(0, view)) {
        std::cerr << "[M5] " << name << " getOutput başarısız\n";
        return false;
    }
    const float* f = static_cast<const float*>(view.data);
    std::cout << "[M5] " << name << " çıktı[0] ilk 3 değer: " << f[0] << ", " << f[1]
              << ", " << f[2] << " (dummy girdi, değerler anlamsız)\n";
    engine.releaseOutputs();

    std::cout << "[M5] " << name << " -> run() BAŞARILI, çökme yok.\n";
    return true;
}

// head_engine_: zf/xf — CPU'da transpose edilmiş float32 tensörler
// (gerçek LightTrackImpl akışının aynısı).
bool runTensorInputs(const std::string& name, RknnInferenceEngine& engine) {
    const auto& inputs = engine.inputProfiles();
    if (inputs.empty()) {
        std::cerr << "[M5] " << name << " girdi profili boş\n";
        return false;
    }

    std::vector<std::vector<float>> dummy_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        dummy_inputs[i].assign(inputs[i].n_elems, 0.0f);
        uint32_t byte_size = static_cast<uint32_t>(dummy_inputs[i].size() * sizeof(float));
        if (!engine.setInputCopy(static_cast<uint32_t>(i), dummy_inputs[i].data(),
                                  byte_size,
                                  RknnInferenceEngine::InputDataType::Float32Tensor)) {
            std::cerr << "[M5] " << name << " girdi[" << i << "] bağlanamadı\n";
            return false;
        }
    }

    if (!engine.run()) {
        std::cerr << "[M5] " << name << " run() başarısız\n";
        return false;
    }

    for (size_t i = 0; i < engine.outputProfiles().size(); ++i) {
        RknnInferenceEngine::OutputView view;
        if (!engine.getOutput(static_cast<uint32_t>(i), view)) {
            std::cerr << "[M5] " << name << " çıktı[" << i << "] alınamadı\n";
            return false;
        }
        const float* f = static_cast<const float*>(view.data);
        std::cout << "[M5] " << name << " çıktı[" << i << "] ilk 3 değer: " << f[0]
                  << ", " << f[1] << ", " << f[2] << " (dummy girdi, değerler anlamsız)\n";
    }
    engine.releaseOutputs();

    std::cout << "[M5] " << name << " -> run() BAŞARILI, çökme yok.\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Kullanım: " << argv[0]
                  << " <template.rknn> <search.rknn> <head.rknn>\n";
        return 1;
    }

    std::cout << "[M5] 3 model, klasik yol + doğru tip beyanı testi başlıyor...\n";

    bool ok = true;

    {
        std::cout << "\n=== template_engine_ (init) ===\n";
        RknnInferenceEngine engine;
        if (engine.initialize(argv[1])) {
            ok &= runImageInput("template_engine_", engine);
        } else {
            ok = false;
        }
        engine.shutdown();
    }
    {
        std::cout << "\n=== search_engine_ (backbone) ===\n";
        RknnInferenceEngine engine;
        if (engine.initialize(argv[2])) {
            ok &= runImageInput("search_engine_", engine);
        } else {
            ok = false;
        }
        engine.shutdown();
    }
    {
        std::cout << "\n=== head_engine_ (neck_head) ===\n";
        RknnInferenceEngine engine;
        if (engine.initialize(argv[3])) {
            ok &= runTensorInputs("head_engine_", engine);
        } else {
            ok = false;
        }
        engine.shutdown();
    }

    std::cout << "\n[M5] "
              << (ok ? "TÜM MODELLER ÇALIŞTI." : "EN AZ BİR MODEL BAŞARISIZ.") << "\n";
    return ok ? 0 : 1;
}