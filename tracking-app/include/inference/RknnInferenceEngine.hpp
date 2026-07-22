#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// 02-IInferenceEngine.md sözleşmesinin somut implementasyonu. Gerçek
// rknn_api.h'a göre yazılmıştır (araştırma raporlarının uydurduğu
// rknn_inputs_map/sync gibi bu SDK'da var olmayan API'ler KULLANILMAZ).
enum class Precision { INT8, FP16, UNKNOWN };

// NPU çekirdek ataması artık otomatik (RKNN_NPU_CORE_AUTO).
// RKNN SDK, modelleri otomatik olarak NPU çekirdeklerine dağıtır.
// Bu, referans LightTrack projesindeki gibi davranmak istediğimiz için
// gerekli bir değişikliktir.

struct TensorProfile {
    uint32_t index = 0;
    std::string name;
    uint32_t n_dims = 0;
    uint32_t dims[4] = {0, 0, 0, 0};
    uint32_t n_elems = 0;
    uint32_t size = 0;
    uint32_t w_stride = 0;
    uint32_t h_stride = 0;
    uint32_t size_with_stride = 0;
    Precision precision = Precision::UNKNOWN;
    int32_t zp = 0;
    float scale = 1.0f;
    int32_t raw_type = 0;  // rknn_tensor_type ham değeri (setInputCopy için)
    int32_t raw_fmt = 0;   // rknn_tensor_format ham değeri (NCHW/NHWC)
};

// Tek bir .rknn modelini saran, precision-farkında, çekirdek ataması ZORUNLU
// wrapper. Sistemde her model (template_backbone, search_backbone, head,
// ileride detector) kendi RknnInferenceEngine örneğine sahip olacak.
class RknnInferenceEngine {
public:
    RknnInferenceEngine();
    ~RknnInferenceEngine();

    RknnInferenceEngine(const RknnInferenceEngine&) = delete;
    RknnInferenceEngine& operator=(const RknnInferenceEngine&) = delete;

    // Auto scheduling: rknn_init ile model yüklenir, RKNN SDK NPU çekirdeklerini
    // otomatik olarak yönetir. Çekirdek ataması yapılmaz (core_mask=0 → AUTO).
    bool initialize(const std::string& model_path);

    const std::vector<TensorProfile>& inputProfiles() const;
    const std::vector<TensorProfile>& outputProfiles() const;

    // Zero-copy bağlama: RGA çıktısı gibi bir DmaBuffer'ın fd'sini doğrudan
    // NPU girişine bağlar (rknn_create_mem_from_fd + rknn_set_io_mem).
    // UYARI: initialize() artık RKNN_FLAG_MEM_ALLOC_OUTSIDE KULLANMIYOR
    // (klasik rknn_inputs_set ile çakıştığı M5'te keşfedildi — bkz. .cpp
    // notu). Bu metod şu an hiçbir yerde çağrılmıyor; gerçekten zero-copy
    // gereken bir senaryo (örn. native formatı uyumlu bir detector modeli)
    // ortaya çıkarsa, o motor için AYRI bir initialize() yolu/flag'i
    // gerekecek — aynı context ikisini karışık desteklemiyor.
    bool bindInputZeroCopy(uint32_t tensor_index, const DmaBufferPtr& buffer);

    // Klasik kopyalı giriş: RGA'dan gelmeyen ya da modelin NATIVE tipiyle
    // (örn. FLOAT16) uyuşmayan ham veri için (örn. RGA'nın ürettiği ham
    // UINT8 piksel, ya da backbone->head arasındaki transpose edilmiş
    // float32 ara tensör). data_type, BİZİM verdiğimiz verinin GERÇEK
    // tipini belirtir — modelin kendi sorguladığı (native) tipini DEĞİL.
    // (M5 testinde keşfedilen kritik ayrım: pass_through=0 iken RKNN'e
    // "bu veri şu tipte" deriz, RKNN kendi native formatına o veriden
    // dönüştürür — modelin native tipini veri tipi sanıp bildirmek,
    // dönüşüm zincirini bozup "invalid tensor malloc size" çökmesine
    // yol açıyordu.)
    //
    // KRİTİK SEMANTİK (M6 canlı testinde keşfedildi — "donmuş bbox"un kök
    // nedeni): bu çağrı veriyi HEMEN set etmez, biriktirir; tüm girişler
    // run() anında TEK rknn_inputs_set çağrısıyla topluca verilir. Çok
    // girişli modellerde (head: zf+xf) tensörleri ayrı rknn_inputs_set
    // çağrılarıyla parça parça set etmek RKNN'de tanımsız davranış —
    // RKNN_SUCC döner ama çıktı sessizce çöp olur. Bu yüzden `data`
    // işaretçisi run() dönene kadar canlı kalmak ZORUNDADIR.
    enum class InputDataType { UInt8Image, Float32Tensor };
    bool setInputCopy(uint32_t tensor_index, const void* data, uint32_t size,
                       InputDataType data_type);

    bool run();

    // want_float HER ZAMAN zorlanır (precision ne olursa olsun) —
    // lighttrack.cpp'nin keşfettiği gerçek riske göre: "float" build'lerde
    // bile rknn-toolkit2 bazen çıktıyı performans için sessizce INT8'e
    // çevirebiliyor; want_float=false kalırsa çağıran bu veriyi (float*)
    // sanıp okur -> bellek dışına taşma / çöp değer.
    struct OutputView {
        const void* data = nullptr;
        uint32_t size = 0;
        TensorProfile profile;
    };
    bool getOutput(uint32_t tensor_index, OutputView& out);
    void releaseOutputs();

    // RKNN_QUERY_PERF_RUN ile NPU'nun KENDİ BİLDİRDİĞİ saf inference
    // süresini (mikrosaniye) döner — want_float dequant maliyeti dahil
    // DEĞİL, sadece donanım çalışma süresi. run()+getOutput() çağrılarından
    // SONRA geçerlidir (rknn_api.h: "valid after rknn_outputs_get").
    // Başarısızsa -1 döner.
    int64_t lastRunDurationUs() const;

    void shutdown();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};