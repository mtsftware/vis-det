#pragma once

#include "tracking/MultiObjectTracker.hpp"
#include "types/DmaBuffer.hpp"
#include "types/VideoCoding.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Referans: tracking-app/include/pipeline/PipelineOrchestrator.hpp ile AYNI
// thread modeli (06-system-architecture.md §5, CLAUDE.md "Canlıda
// Doğrulanmış Kurallar" §5): Ingestion (GStreamer) -> MppDecoder (kendi
// put/get thread'leri) -> [EN-YENİ-KARE SLOTU, decode ASLA bloklanmaz, bayat
// kare düşürülür] -> TEK seri işleme thread'i.
//
// tıkanma/takılma teşhisi: main.cpp'nin önceki sürümü tüm pipeline'ı
// (letterbox+NPU+postprocess+çizim+stream) MppDecoder'ın kendi get_thread'i
// İÇİNDE senkron çalıştırıyordu — decode thread bir sonraki kareyi
// çekemeden pipeline'ın bitmesini bekliyordu, MPP'nin 16 slotluk decode
// havuzu dolunca decode durup RTSP tarafında görünür takılmaya yol açıyordu.
// Bu sınıf decode'u işleme thread'inden TAMAMEN ayırır: decode callback'i
// sadece en-yeni-kareyi bir slota bırakıp döner, ağır iş ayrı thread'de.
//
// Farklar (tracking-app'e göre): ITracker yerine YOLOv8n detection + NMS +
// MultiObjectTracker; ITracker RGA/RKNN detaylarını gizliyordu, burada
// çizim de (postprocess'in bir parçası, önceki faz kararı) aynı thread'de
// — ResultCallback SADECE streamer.pushFrame() gibi hafif bir iş yapmalı.
struct PipelineConfig {
    std::string rtsp_url;
    uint32_t rtsp_latency_ms = 5000;
    VideoCoding coding = VideoCoding::H264;
    // airockchip RKNN-optimised YOLOv8 export, COCO 80-sinif, 9 cikti
    // tensoru (edge-ai-workshop-rknn/yolov8n_int8.rknn ile AYNI dosya) —
    // eski tek-tensor best-rk3588.rknn'in YERINE.
    std::string model_path = "models/yolov8n_int8.rknn";
    std::string labels_path = "models/coco_labels.txt";
    uint32_t model_width = 640;
    uint32_t model_height = 640;
    float detection_conf_thresh = 0.4f;
    float nms_iou_thresh = 0.45f;
};

struct FrameResult {
    DmaBufferPtr frame;  // cizim TAMAMLANMIS, stream'e hazir (NV12/NV16)
    std::vector<TrackableObject> tracked_objects;
    int detection_count = 0;
    uint64_t frame_index = 0;
};

class PipelineOrchestrator {
public:
    // İŞLEME THREAD'İNDE çağrılır — hafif tutulmalı (streamer.pushFrame
    // gibi). Ağır iş burada koşarsa işleme FPS'i ona kilitlenir (decode
    // ETKİLENMEZ — en-yeni-kare slotu deseni decode'u zaten izole ediyor).
    using ResultCallback = std::function<void(const FrameResult&)>;

    explicit PipelineOrchestrator(PipelineConfig config);
    ~PipelineOrchestrator();

    PipelineOrchestrator(const PipelineOrchestrator&) = delete;
    PipelineOrchestrator& operator=(const PipelineOrchestrator&) = delete;

    void setResultCallback(ResultCallback cb);

    // Modeli yukler, demuxer+decoder'i kurar/baslatir, isleme thread'ini acar.
    bool start();
    void stop();

    struct Stats {
        uint64_t frames_decoded = 0;
        uint64_t frames_processed = 0;
        uint64_t frames_dropped = 0;  // en-yeni-kare slotu doluyken gelen (bayat) kareler
        uint32_t active_tracks = 0;
        uint32_t total_tracked = 0;
        uint32_t lost_tracks = 0;
    };
    Stats getStats() const;

    // İlk decode edilen karenin gerçek boyutunu/formatını bilmek isteyen
    // çağıranlar (örn. RtspStreamer::setDimensions) için.
    bool firstFrameSeen() const;
    uint32_t sourceWidth() const;
    uint32_t sourceHeight() const;
    PixelFormat sourceFormat() const;

    // coco_labels.txt icerigi (start() sirasinda yuklenir). class_id bu
    // vektorun bir indeksidir; sinir disi/yuklenemediyse bos liste doner
    // (cagiran numerik ID'ye geri duser).
    const std::vector<std::string>& labels() const;

    static void pinCurrentThreadToBigCores();
    static void restoreCurrentThreadAllCores();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};
