#pragma once

#include "types/DmaBuffer.hpp"

#include <cstdint>
#include <memory>

// KRİTİK MİMARİ KURALLAR (Rapor 2, Rapor 3):
// - immakeBorder() KESİNLİKLE KULLANILMAZ (RGA2 + DMA32/IOMMU çökme riski,
//   üretim ortamlarında doğrulanmış bir hata sınıfı).
// - Letterbox iki RGA çağrısıyla kurulur: imfill (gri zemin) + improcess
//   (kaynağı ölçekle, merkeze yerleştir).
// - Kaynak formatı (NV12/NV16/...) ASLA varsayılmaz — decode'dan gelen
//   gerçek format çağırana açıkça parametre olarak verilir (M4'teki NV16
//   sürprizinden ders: format hep doğrulanmalı, tahmin edilmemeli).
enum class RgaPixelFormat { NV12, NV16, RGB888, BGR888 };

struct LetterboxResult {
    double ratio = 1.0;
    int dst_offset_x = 0;
    int dst_offset_y = 0;
    int scaled_width = 0;
    int scaled_height = 0;
};

class RgaPreprocessor {
public:
    RgaPreprocessor();
    ~RgaPreprocessor();

    RgaPreprocessor(const RgaPreprocessor&) = delete;
    RgaPreprocessor& operator=(const RgaPreprocessor&) = delete;

    // target_format: NV12 -> görsel doğrulama (RtspStreamer ile basılabilir).
    //                RGB888/BGR888 -> gerçek model girişi (henüz test edilmedi).
    bool configure(uint32_t target_width, uint32_t target_height,
                   RgaPixelFormat target_format);

    // YAŞAM DÖNGÜSÜ SÖZLEŞMESİ (M7'de güncellendi): process/cropTargetCentric/
    // flipHorizontal'ın döndürdüğü DmaBuffer'lar artık iç DmaBufferPool'dan
    // (04-IBufferPool.md) GERÇEK referans sayımıyla gelir — M6'daki geçici
    // "aynı boyutta 3 çağrı geçerli" ring kısıtı KALKTI. Çağıran buffer'ı
    // istediği kadar tutabilir; son shared_ptr düştüğünde slot havuza
    // otomatik döner. Tek dikkat: aynı boyuttan 4 buffer'ı AYNI ANDA tutmak
    // havuzu tüketir, üretici 500ms zaman aşımıyla hata loglar (kasıtlı
    // geri basınç — sızıntıyı sessiz kilitlenme yerine log yapar).
    //
    // source: MPP decode çıktısı. source_format: gerçek format (decode'un
    // mpp_frame_get_fmt() ile doğruladığı format, ASLA varsayılmaz).
    bool process(const DmaBufferPtr& source, RgaPixelFormat source_format,
                 DmaBufferPtr& out_buffer, LetterboxResult& out_transform);

    // Hedef-merkezli bağlam kırpma (Siamese tracker crop). LightTrack
    // referans kodundaki get_subwindow_tracking()'in RGA karşılığı —
    // matematik (context hesabı, pad miktarları) AYNI, sadece piksel
    // işlemi (pad+crop+resize) RGA'da. Çıktı her zaman RGB888 (tracker
    // modelleri RGB bekliyor).
    // center_x/y: hedefin son bilinen piksel konumu (kaynak görüntüde).
    // context_size: kırpılacak kare bölgenin kenar uzunluğu (s_z/s_x).
    // model_size: hedef model girişi (127=exemplar, 288=instance).
    // pad_r/g/b: görüntü sınırı dışına taşan alanlar için doldurma rengi.
    bool cropTargetCentric(const DmaBufferPtr& source, RgaPixelFormat source_format,
                            float center_x, float center_y, int context_size,
                            int model_size, uint8_t pad_r, uint8_t pad_g,
                            uint8_t pad_b, DmaBufferPtr& out_buffer);

    // Yatay çevirme (RGA imflip ile, donanımsal). Template augmentasyonu
    // için kullanılır. Kaynak RGB888 olmalı (zaten cropTargetCentric
    // çıktısı bu formatta).
    bool flipHorizontal(const DmaBufferPtr& source, DmaBufferPtr& out_buffer);

    // Kaynakla AYNI boyut/format'ta, havuzdan alınmış YENİ bir tampona
    // donanımsal (RGA improcess) piksel kopyası. ÖNEMLİ KULLANIM AMACI:
    // MPP kod çözücünün çıktı tamponu (main_m8/m9_test.cpp'deki
    // drawBboxOutline'ın hedefi), MPP_DEC_SET_EXT_BUF_GROUP ile paylaşılan
    // sınırlı bir havuzdan geliyor ve kod çözücü bu AYNI fiziksel belleği
    // sonraki P-frame'ler için referans kare (DPB) olarak da kullanıyor
    // olabilir — decode çıktısına yerinde (in-place) çizim, VPU'nun o
    // belleği hâlâ referans olarak okuduğu bir anla çakışırsa görüntüye
    // kalıcı bozulma/"yapışma" olarak yansıyabilir (NanoTrack'in hızlı
    // backbone'uyla canlı testte gözlemlendi — LightTrack'in yavaş
    // backbone'u çakışma penceresini doğal olarak daraltıyordu, hata
    // gizliydi). Çözüm: çizimi HER ZAMAN bu fonksiyonun döndürdüğü
    // bağımsız kopya üzerinde yap, kod çözücünün kendi tamponuna asla
    // yazma. source_format: source->format ile TUTARLI olmalı (main'de
    // zaten drawBboxOutline'ın kullandığı aynı PixelFormat->RgaPixelFormat
    // dönüşümüyle üretiliyor).
    bool cloneFrame(const DmaBufferPtr& source, RgaPixelFormat source_format,
                     DmaBufferPtr& out_buffer);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};