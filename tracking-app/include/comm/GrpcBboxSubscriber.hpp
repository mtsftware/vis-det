#pragma once

#include "tracking/ITracker.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// M11: proto/tracking.proto (DroneObjectTrackingService) ile üretilen kod
// (generated/tracking.grpc.pb.h) bu header'ı dahil eden .cpp dosyasında
// görünür olmalı — burada ileri bildirim kullanılmıyor çünkü üretilen
// tipler Impl pimpl-idiom arkasında saklanıp yalnızca
// GrpcBboxSubscriber.cpp içinde bilinir.
//
// Board, GCS'e (server) bağlanan gRPC CLIENT'tır. Stream (server-streaming)
// RPC'sini çağırır; GCS'te yeni bir BBox seçildiğinde stream üzerinden
// okur. TargetPoint gönderimi ŞİMDİLİK
// KALDIRILDI — bu sınıf sadece tek yönlü (GCS -> board) BBox akışını
// yönetir; M10'daki UartMavlinkLink'in (BBOX alım) karşılığıdır.
// BboxCallback sözleşmesi UartMavlinkLink ile birebir aynı.
class GrpcBboxSubscriber {
public:
    using BboxCallback = std::function<void(const BBox&)>;

    // host/port: GCS tarafındaki gRPC server adresi (örn. "192.168.1.50", 50051).
    GrpcBboxSubscriber(std::string host, uint16_t port);
    ~GrpcBboxSubscriber();

    GrpcBboxSubscriber(const GrpcBboxSubscriber&) = delete;
    GrpcBboxSubscriber& operator=(const GrpcBboxSubscriber&) = delete;

    void setBboxCallback(BboxCallback cb);

    // Kanalı açar ve arka planda abonelik/okuma thread'ini başlatır. Thread
    // bağlantı koparsa (Read() false döner) otomatik yeniden abone olur —
    // start() sadece thread'i başlatır, ilk bağlantının kurulmasını beklemez.
    bool start();
    void stop();

    struct Stats {
        uint64_t bbox_received = 0;
        uint64_t stream_reconnects = 0;
    };
    Stats getStats() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};