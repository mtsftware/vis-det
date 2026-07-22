#include "comm/GrpcBboxSubscriber.hpp"

#include "generated/tracking.grpc.pb.h"

#include <google/protobuf/empty.pb.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace {
constexpr auto kReconnectDelay = std::chrono::milliseconds(1000);
}  // namespace

struct GrpcBboxSubscriber::Impl {
    std::string host;
    uint16_t port;
    BboxCallback bbox_cb;

    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<drone::object_tracking::DroneObjectTrackingService::Stub> stub;

    std::thread conn_thread;
    std::atomic<bool> running{false};

    // context sadece connectionLoop() tarafından yazılır, stop() tarafından
    // TryCancel() için okunur -- aradaki senkronizasyon ctx_mtx ile sağlanır.
    std::mutex ctx_mtx;
    std::unique_ptr<grpc::ClientContext> context;

    std::atomic<uint64_t> bbox_received{0};
    std::atomic<uint64_t> stream_reconnects{0};

    Impl(std::string h, uint16_t p) : host(std::move(h)), port(p) {}

    void connectionLoop() {
        bool first_attempt = true;
        while (running.load()) {
            auto ctx = std::make_unique<grpc::ClientContext>();
            google::protobuf::Empty empty_req;
            auto reader = stub->Stream(ctx.get(), empty_req);

            {
                std::lock_guard<std::mutex> lock(ctx_mtx);
                context = std::move(ctx);
            }

            std::cout << "[GrpcBboxSubscriber] " << host << ":" << port
                      << " Stream ile abone olundu\n";

            if (!first_attempt) {
                stream_reconnects.fetch_add(1, std::memory_order_relaxed);
            }
            first_attempt = false;

            drone::object_tracking::BBox msg;
            while (running.load() && reader->Read(&msg)) {
                BBox bbox;
                bbox.x = static_cast<int>(msg.x());
                bbox.y = static_cast<int>(msg.y());
                bbox.width = static_cast<int>(msg.w());
                bbox.height = static_cast<int>(msg.h());

                if (bbox.width <= 0 || bbox.height <= 0) {
                    std::cerr << "[GrpcBboxSubscriber] geçersiz BBox (w/h <= 0), "
                                 "yok sayıldı\n";
                    continue;
                }

                bbox_received.fetch_add(1, std::memory_order_relaxed);
                std::cout << "[GrpcBboxSubscriber] BBox alındı: (" << bbox.x << ","
                          << bbox.y << "," << bbox.width << "," << bbox.height << ")\n";
                if (bbox_cb) bbox_cb(bbox);
            }

            grpc::Status status = reader->Finish();
            {
                std::lock_guard<std::mutex> lock(ctx_mtx);
                context.reset();
            }

            if (!running.load()) break;

            std::cerr << "[GrpcBboxSubscriber] stream koptu ("
                      << status.error_message() << "), "
                      << kReconnectDelay.count() << "ms sonra yeniden abone "
                                                     "olunacak\n";
            std::this_thread::sleep_for(kReconnectDelay);
        }
    }
};

GrpcBboxSubscriber::GrpcBboxSubscriber(std::string host, uint16_t port)
    : impl_(std::make_unique<Impl>(std::move(host), port)) {}

GrpcBboxSubscriber::~GrpcBboxSubscriber() { stop(); }

void GrpcBboxSubscriber::setBboxCallback(BboxCallback cb) {
    impl_->bbox_cb = std::move(cb);
}

bool GrpcBboxSubscriber::start() {
    if (impl_->running.load()) return true;

    std::ostringstream target;
    target << impl_->host << ":" << impl_->port;
    impl_->channel =
        grpc::CreateChannel(target.str(), grpc::InsecureChannelCredentials());
    impl_->stub = drone::object_tracking::DroneObjectTrackingService::NewStub(impl_->channel);

    impl_->running.store(true);
    impl_->conn_thread = std::thread([this] { impl_->connectionLoop(); });
    std::cout << "[GrpcBboxSubscriber] " << target.str() << " için abonelik "
                 "thread'i başlatıldı\n";
    return true;
}

void GrpcBboxSubscriber::stop() {
    if (!impl_->running.exchange(false)) return;

    {
        // TryCancel, connectionLoop()'un o an bloklu olduğu Read() çağrısını
        // hemen kırar -- aksi halde GCS tarafı stream'i kapatana kadar
        // join() sonsuza kadar bekleyebilir.
        std::lock_guard<std::mutex> lock(impl_->ctx_mtx);
        if (impl_->context) impl_->context->TryCancel();
    }

    if (impl_->conn_thread.joinable()) impl_->conn_thread.join();
}

GrpcBboxSubscriber::Stats GrpcBboxSubscriber::getStats() const {
    Stats s;
    s.bbox_received = impl_->bbox_received.load(std::memory_order_relaxed);
    s.stream_reconnects = impl_->stream_reconnects.load(std::memory_order_relaxed);
    return s;
}