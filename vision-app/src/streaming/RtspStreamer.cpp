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

struct RtspStreamer::Impl {
    uint16_t port;
    std::string mount_path;
    uint32_t width = 0;   // logical (caps'te kullanilan, hizalanmamis)
    uint32_t height = 0;  // logical (caps'te kullanilan, hizalanmamis)
    uint32_t aligned_w = 0;  // width 16'ya hizalanmis (stride olarak kullanilir)
    uint32_t aligned_h = 0;  // height 16'ya hizalanmis
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

/* Sinyal callback */
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
    impl_->aligned_w = ((width + 15) / 16) * 16;
    impl_->aligned_h = ((height + 15) / 16) * 16;
    impl_->fps = fps;

    static std::once_flag gst_init_flag;
    std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });
}

RtspStreamer::~RtspStreamer() = default;

// Boyutlari caltisma zamaninda guncelle (dynamic resize icin)
void RtspStreamer::setDimensions(uint32_t width, uint32_t height) {
    impl_->width = width;
    impl_->height = height;
    impl_->aligned_w = ((width + 15) / 16) * 16;
    impl_->aligned_h = ((height + 15) / 16) * 16;
}

bool RtspStreamer::start() {
    impl_->server = gst_rtsp_server_new();
    std::string port_str = std::to_string(impl_->port);
    g_object_set(impl_->server, "service", port_str.c_str(), nullptr);

    GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(impl_->server);
    impl_->factory = gst_rtsp_media_factory_new();

    // caps: logical (hizalanmamis) w/h — mpph264enc logical degerleri bekler
    std::string launch =
        "( appsrc name=mysrc is-live=true format=time do-timestamp=true "
        "caps=\"video/x-raw,format=NV12,width=" + std::to_string(impl_->width) +
        ",height=" + std::to_string(impl_->height) +
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
              << impl_->mount_path << " yayinda (NV12, "
              << impl_->width << "x" << impl_->height << ").\n";
    return true;
}

void RtspStreamer::stop() {
    if (impl_) impl_->stop();
}

/* pushFrame(): tracking-app/src/streaming/RtspStreamer.cpp:145-202 ile BIREBIR AYNI
 * - Y plane: tek memcpy (w*h byte) — stride yok
 * - UV plane: NV16 her 2 satirindan 1 (nearest-neighbor downsample)
 * - mpph264enc her zaman NV12 bekler (caps format=NV12)
 */
void RtspStreamer::pushFrame(const DmaBufferPtr& frame) {
    std::lock_guard<std::mutex> lock(impl_->appsrc_mutex);
    if (!impl_->current_appsrc || !frame || !frame->virt_addr) {
        return;
    }

    // tracking-app satir 161-165: logical width/height kullan (stride degil)
    const size_t y_size = static_cast<size_t>(impl_->width) * impl_->height;
    const size_t nv16_uv_row_bytes = impl_->width;  // W byte (W/2 U + W/2 V interleaved)
    const size_t nv12_uv_size = y_size / 2;
    const size_t nv12_total = y_size + nv12_uv_size;

    // tracking-app satir 167: güvenlik kontrolü
    if (frame->size < y_size + nv16_uv_row_bytes * impl_->height) {
        return;
    }

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, nv12_total, nullptr);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(frame->virt_addr);
    uint8_t* dst = map.data;

    // tracking-app satir 179: Y plane TAM KOPYA (tek memcpy, stride yok)
    std::memcpy(dst, src, y_size);

    // tracking-app satir 184-190: UV plane — her 2 satirdan 1 (NV16→NV12 downsample)
    const uint8_t* src_uv = src + y_size;
    uint8_t* dst_uv = dst + y_size;
    for (uint32_t row = 0; row < impl_->height / 2; ++row) {
        const uint8_t* src_row = src_uv + (static_cast<size_t>(row) * 2) * nv16_uv_row_bytes;
        uint8_t* dst_row = dst_uv + static_cast<size_t>(row) * nv16_uv_row_bytes;
        std::memcpy(dst_row, src_row, nv16_uv_row_bytes);
    }

    gst_buffer_unmap(buffer, &map);

    // tracking-app satir 194-197: gst_app_src_push_buffer
    GstFlowReturn ret =
        gst_app_src_push_buffer(GST_APP_SRC(impl_->current_appsrc), buffer);

    if (ret == GST_FLOW_OK) {
        impl_->frames_pushed.fetch_add(1, std::memory_order_relaxed);
    }
}

/* Y planini row-by-row kopyala (stride-aware) */
static void copyYPlaneRowByRow(const uint8_t* src, uint8_t* dst,
                                uint32_t src_stride, uint32_t dst_width, uint32_t dst_height) {
    for (uint32_t y = 0; y < dst_height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dst_width,
                    src + static_cast<size_t>(y) * src_stride,
                    static_cast<size_t>(dst_width));
    }
}

void RtspStreamer::pushRawNV12(const DmaBufferPtr& frame) {
    std::lock_guard<std::mutex> lock(impl_->appsrc_mutex);
    if (!impl_->current_appsrc || !frame || !frame->virt_addr) {
        return;
    }

    const uint32_t lw = impl_->width;      // logical (caps/degerler)
    const uint32_t lh = impl_->height;     // logical (caps/degerler)
    const uint32_t sw = frame->w_stride ? frame->w_stride : lw;  // kaynak stride
    const uint32_t sh = frame->h_stride ? frame->h_stride : lh;  // kaynak stride

    // NV12 tightly-packed boyut: W*H (Y) + W*H/2 (UV)
    const size_t y_size = static_cast<size_t>(lw) * lh;
    const size_t uv_size = y_size / 2;
    const size_t total_size = y_size + uv_size;

    if (!frame->virt_addr || frame->fd < 0) {
        return;
    }

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, total_size, nullptr);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(frame->virt_addr);
    uint8_t* dst = map.data;

    // Y planini: kaynak stride ile oku, logikal hizalama ile yaz
    for (uint32_t y = 0; y < lh; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * lw,
                    src + static_cast<size_t>(y) * sw,
                    lw);
    }

    // UV planini: kaynak UV, sh/2 satir, her satir sw byte
    const size_t src_uv_offset = static_cast<size_t>(sh) * sw;
    const uint8_t* src_uv = src + src_uv_offset;
    uint8_t* dst_uv = dst + y_size;
    for (uint32_t y = 0; y < lh / 2; ++y) {
        std::memcpy(dst_uv + static_cast<size_t>(y) * lw,
                    src_uv + static_cast<size_t>(y) * sw,
                    lw);
    }

    gst_buffer_unmap(buffer, &map);

    GstFlowReturn ret =
        gst_app_src_push_buffer(GST_APP_SRC(impl_->current_appsrc), buffer);

    if (ret == GST_FLOW_OK) {
        impl_->frames_pushed.fetch_add(1, std::memory_order_relaxed);
    }
}

void RtspStreamer::pushRawNV16(const DmaBufferPtr& frame) {
    std::lock_guard<std::mutex> lock(impl_->appsrc_mutex);
    if (!impl_->current_appsrc || !frame || !frame->virt_addr) {
        return;
    }

    const uint32_t lw = impl_->width;      // logical (no padding)
    const uint32_t lh = impl_->height;     // logical (no padding)
    const uint32_t src_w = frame->w_stride ? frame->w_stride : lw;

    // NV16 boyutu: W*H*2 (YUV 4:2:2, her piksel 2 byte)
    const size_t total_size = static_cast<size_t>(lw) * lh * 2;

    if (frame->size < total_size) {
        return;
    }

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, total_size, nullptr);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(frame->virt_addr);
    uint8_t* dst = map.data;

    // NV16 her pixel 2 byte: row-by-row kopyala
    const size_t src_row_bytes = static_cast<size_t>(src_w) * 2;
    const size_t dst_row_bytes = static_cast<size_t>(lw) * 2;

    for (uint32_t y = 0; y < lh; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dst_row_bytes,
                    src + static_cast<size_t>(y) * src_row_bytes,
                    dst_row_bytes);
    }

    gst_buffer_unmap(buffer, &map);

    GstFlowReturn ret =
        gst_app_src_push_buffer(GST_APP_SRC(impl_->current_appsrc), buffer);

    if (ret == GST_FLOW_OK) {
        impl_->frames_pushed.fetch_add(1, std::memory_order_relaxed);
    }
}

void RtspStreamer::pushRawRGB888(const DmaBufferPtr& frame) {
    std::lock_guard<std::mutex> lock(impl_->appsrc_mutex);
    if (!impl_->current_appsrc || !frame || !frame->virt_addr) {
        return;
    }

    const uint32_t lw = impl_->width;
    const uint32_t lh = impl_->height;
    const uint32_t src_w = frame->w_stride ? frame->w_stride : lw;

    size_t frame_size = static_cast<size_t>(lw) * lh * 3;
    size_t copy_size = std::min(frame_size, static_cast<size_t>(frame->size));

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame_size, nullptr);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return;
    }

    // RGB888 row-by-row stride-aware copy
    const uint8_t* src = static_cast<const uint8_t*>(frame->virt_addr);
    uint8_t* dst = map.data;
    const size_t src_row = static_cast<size_t>(src_w) * 3;
    const size_t dst_row = static_cast<size_t>(lw) * 3;

    for (uint32_t y = 0; y < lh; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dst_row,
                    src + static_cast<size_t>(y) * src_row,
                    dst_row);
    }

    gst_buffer_unmap(buffer, &map);

    GstFlowReturn ret =
        gst_app_src_push_buffer(GST_APP_SRC(impl_->current_appsrc), buffer);

    if (ret == GST_FLOW_OK) {
        impl_->frames_pushed.fetch_add(1, std::memory_order_relaxed);
    }
}