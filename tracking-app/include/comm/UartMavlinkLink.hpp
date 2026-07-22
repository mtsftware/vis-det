#pragma once

#include "tracking/ITracker.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// CLAUDE.md Faz 1 §3: /dev/ttyS8 üzerinde arka planda MAVLINK BBOX mesajı
// bekleyen dinleyici. Kendi thread'inde byte-byte mavlink::mavlink_parse_char
// çalıştırır (mavlink/mavlink_helpers.h) — pipeline'ın tracking thread'ini
// HİÇ bloklamaz, sadece BBOX geldiğinde BboxCallback ile haber verir.
// Aktivasyon kararı (PipelineOrchestrator::requestInitialize çağrısı)
// çağıranın işi — bu sınıf sadece UART/MAVLINK katmanını bilir.
class UartMavlinkLink {
public:
    using BboxCallback = std::function<void(const BBox&)>;

    // device: örn. "/dev/ttyS8" (board) ya da "/dev/ttyUSB0" (host).
    // baud: yaygın MAVLink seri hızlarından biri (57600 varsayılan).
    explicit UartMavlinkLink(std::string device, uint32_t baud = 57600);
    ~UartMavlinkLink();

    UartMavlinkLink(const UartMavlinkLink&) = delete;
    UartMavlinkLink& operator=(const UartMavlinkLink&) = delete;

    void setBboxCallback(BboxCallback cb);

    // Portu açar (termios raw/8N1) ve okuma thread'ini başlatır.
    bool start();
    void stop();

    struct Stats {
        uint64_t bytes_read = 0;
        uint64_t messages_parsed = 0;
        uint64_t bbox_received = 0;
        uint64_t parse_errors = 0;
    };
    Stats getStats() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};