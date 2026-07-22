#pragma once

#include <cstdint>
#include <memory>
#include <string>

// CLAUDE.md Faz 1 §4: inference çıktısından çıkarılan Target Point'i UDP ile
// (varsayılan olarak RTSP kaynağıyla aynı host'a) MAVLINK TARGET_POINT
// mesajı olarak gönderir. Bağlantısız/best-effort — telemetri amaçlı,
// gönderim başarısızlığı pipeline'ı durdurmaz.
class UdpTargetPointSender {
public:
    UdpTargetPointSender(std::string host, uint16_t port);
    ~UdpTargetPointSender();

    UdpTargetPointSender(const UdpTargetPointSender&) = delete;
    UdpTargetPointSender& operator=(const UdpTargetPointSender&) = delete;

    bool start();
    void stop();

    // x,y: hedef noktanın piksel koordinatları (bbox merkezi).
    void send(float x, float y);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};