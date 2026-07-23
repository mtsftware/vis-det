#include "streaming/RtspStreamer.hpp"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

/* Impl sinifi */
struct RtspStreamer::Impl {
    uint16_t port;
    std::string mount_path;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 30;

    GstRTSPServer* server = nullptr;
    GstRTSPMediaFactory* factory = nullptr;
    guint source_id = 0;

    GMainLoop* loop = nullptr;
    std::thread loop_thread;

    std::mutex appsrc_mutex;
    GstElement* current_appsrc = nullptr;

    std::atomic<uint64_t> frames_pushed{0};

    ~Impl() { stop(); }

    void stop() {
        if (loop && g_main_loop_is_running(loop)) {
            g_main_loop_quit(loop);
        }
        if (loop_thread.joinable()) loop_thread.join();
        if (source_id) {
            g_source_remove(source_id);
            source_id = 0;
        }
        if (server) {
            gst_object_unref(server);
            server = nullptr;
        }
        if (loop) {
            g_main_loop_unref(loop);
            loop = nullptr;
        }
        std::lock_guard<std::mutex> lock(appsrc_mutex);
        current_appsrc = nullptr;
    }
};

/* Sinyal callback — Impl public oldugu icin doğrudan ulasabiliriz */
static void onMediaConfigureCb(GstRTSPMediaFactory* /*factory*/,
                               GstRTSPMedia* media,
                               gpointer user_data) {
    auto* impl = static_cast<RtspStreamer::Impl*>(user_data);
    if (!impl || !media) return;

    GstElement* element = gst_rtsp_media_get_element(media);
    if (!element) return;

    GstElement* appsrc = gst_bin_get_by_name(GST_BIN(element), "mysrc");
    if (appsrc) {
        std::lock_guard<std::mutex> lock(impl->appsrc_mutex);
        impl->current_appsrc = appsrc;
        std::cout << "[RtspStreamer] Client baglandi, appsrc hazir.\n";
    }

    gst_object_unref(element);
}

RtspStreamer::RtspStreamer(uint16_t port, std::string mount_path, uint32_t width,
                           uint32_t height, uint32_t fps)
    : impl_(std::make_unique<Impl>()) {
    impl_->port = port;
    impl_->mount_path = std::move(mount_path);
    impl_->width = width;
    impl_->height = height;
    impl_->fps = fps;

    static std::once_flag gst_init_flag;
    std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });
}

RtspStreamer::~RtspStreamer() = default;

bool RtspStreamer::start() {
    impl_->server = gst_rtsp_server_new();
    std::string port_str = std::to_string(impl_->port);
    g_object_set(impl_->server, "service", port_str.c_str(), nullptr);

    GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(impl_->server);
    impl_->factory = gst_rtsp_media_factory_new();

    std::string launch =
        "( appsrc name=mysrc is-live=true format=time do-timestamp=true "
        "caps=\"video/x-raw,format=NV12,width=" +
        std::to_string(impl_->width) + ",height=" + std::to_string(impl_->height) +
        ",framerate=" + std::to_string(impl_->fps) +
        "/1\" "
        "! mpph264enc ! rtph264pay name=pay0 pt=96 config-interval=1 )";

    gst_rtsp_media_factory_set_launch(impl_->factory, launch.c_str());
    gst_rtsp_media_factory_set_shared(impl_->factory, TRUE);

    g_signal_connect(impl_->factory, "media-configure", G_CALLBACK(onMediaConfigureCb),
                     impl_.get());

    gst_rtsp_mount_points_add_factory(mounts, impl_->mount_path.c_str(),
                                       impl_->factory);
    g_object_unref(mounts);

    impl_->source_id = gst_rtsp_server_attach(impl_->server, nullptr);
    if (impl_->source_id == 0) {
        std::cerr << "[RtspStreamer] Server attach basarisiz\n";
        return false;
    }

    impl_->loop = g_main_loop_new(nullptr, FALSE);
    impl_->loop_thread = std::thread([this] { g_main_loop_run(impl_->loop); });

    std::cout << "[RtspStreamer] rtsp://<board-ip>:" << impl_->port
              << impl_->mount_path << " yayinda.\n";
    return true;
}

void RtspStreamer::stop() {
    if (impl_) impl_->stop();
}

void RtspStreamer::pushRawNV12(const DmaBufferPtr& frame) {
    std::lock_guard<std::mutex> lock(impl_->appsrc_mutex);
    if (!impl_->current_appsrc || !frame || !frame->virt_addr) {
        return;
    }

    size_t frame_size = static_cast<size_t>(impl_->width) * impl_->height * 3 / 2;
    size_t copy_size = std::min(frame_size, static_cast<size_t>(frame->size));

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame_size, nullptr);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return;
    }
    std::memcpy(map.data, frame->virt_addr, copy_size);
    gst_buffer_unmap(buffer, &map);

    GstFlowReturn ret =
        gst_app_src_push_buffer(GST_APP_SRC(impl_->current_appsrc), buffer);

    if (ret == GST_FLOW_OK) {
        impl_->frames_pushed.fetch_add(1, std::memory_order_relaxed);
    }
}