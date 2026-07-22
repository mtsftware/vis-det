#include "ingestion/RtspDemuxer.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>

struct RtspDemuxer::Impl {
    std::string url;
    uint32_t latency_ms = 100;

    GstElement* pipeline = nullptr;
    GstElement* appsink = nullptr;
    GMainLoop* loop = nullptr;
    std::thread loop_thread;
    guint bus_watch_id = 0;

    PacketCallback callback;
    std::atomic<uint64_t> packets_received{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> packets_dropped{0};

    ~Impl() { stop(); }

    void stop() {
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        if (loop && g_main_loop_is_running(loop)) {
            g_main_loop_quit(loop);
        }
        if (loop_thread.joinable()) {
            loop_thread.join();
        }
        if (bus_watch_id) {
            g_source_remove(bus_watch_id);
            bus_watch_id = 0;
        }
        if (pipeline) {
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
        if (loop) {
            g_main_loop_unref(loop);
            loop = nullptr;
        }
        appsink = nullptr;
    }
};

namespace {

GstFlowReturn onNewSample(GstAppSink* sink, gpointer user_data) {
    auto* impl = static_cast<RtspDemuxer::Impl*>(user_data);

    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) {
        // appsink kapatılıyor olabilir (EOS), hata değil.
        return GST_FLOW_OK;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        EncodedPacket packet;
        packet.data.assign(map.data, map.data + map.size);
        packet.pts_ns = GST_BUFFER_PTS_IS_VALID(buffer)
                            ? static_cast<uint64_t>(GST_BUFFER_PTS(buffer))
                            : 0;
        packet.arrival_ns = static_cast<uint64_t>(g_get_monotonic_time()) * 1000ULL;

        gst_buffer_unmap(buffer, &map);

        impl->packets_received.fetch_add(1, std::memory_order_relaxed);
        impl->bytes_received.fetch_add(packet.data.size(), std::memory_order_relaxed);

        if (impl->callback) {
            impl->callback(std::move(packet));
        }
    } else {
        impl->packets_dropped.fetch_add(1, std::memory_order_relaxed);
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

gboolean onBusMessage(GstBus* /*bus*/, GstMessage* msg, gpointer user_data) {
    auto* impl = static_cast<RtspDemuxer::Impl*>(user_data);

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(msg, &err, &debug);
            std::cerr << "[RtspDemuxer] HATA: " << (err ? err->message : "bilinmiyor");
            if (debug) std::cerr << " (" << debug << ")";
            std::cerr << "\n";
            if (err) g_error_free(err);
            if (debug) g_free(debug);
            if (impl->loop) g_main_loop_quit(impl->loop);
            break;
        }
        case GST_MESSAGE_EOS:
            std::cerr << "[RtspDemuxer] Akış sonlandı (EOS)\n";
            if (impl->loop) g_main_loop_quit(impl->loop);
            break;
        case GST_MESSAGE_WARNING: {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_warning(msg, &err, &debug);
            std::cerr << "[RtspDemuxer] UYARI: " << (err ? err->message : "") << "\n";
            if (err) g_error_free(err);
            if (debug) g_free(debug);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

}  // namespace

RtspDemuxer::RtspDemuxer(std::string rtsp_url, uint32_t latency_ms)
    : impl_(std::make_unique<Impl>()) {
    impl_->url = std::move(rtsp_url);
    impl_->latency_ms = latency_ms;

    static std::once_flag gst_init_flag;
    std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });
}

RtspDemuxer::~RtspDemuxer() { stop(); }

void RtspDemuxer::setPacketCallback(PacketCallback cb) {
    impl_->callback = std::move(cb);
}

bool RtspDemuxer::start() {
    // protocols=tcp: paket kaybını UDP'ye göre azaltır, RTP yeniden
    // sıralama sorunlarını basitleştirir. Ağ sağlıklıysa udp de denenebilir.
    // config-interval=1: h264parse, SPS/PPS'i periyodik tekrar eklesin —
    // decoder'ın (M2) her an akışa "temiz" başlayabilmesi için önemli.
    // KRİTİK: h264parse sonrası stream-format=byte-stream,alignment=au
    // ZORUNLU olarak caps ile belirtiliyor. h264parse varsayılan olarak
    // AVC formatı (uzunluk-öncüllü NAL, start code yok) üretebilir; MPP
    // ise Annex-B (00 00 00 01 start code'lu) bekler. Bu caps zorlaması
    // olmadan decode_put_packet "başarılı" döner ama decoder hiç frame
    // üretmez (sessiz format uyuşmazlığı).
    std::string pipeline_desc =
        "rtspsrc location=" + impl_->url +
        " latency=" + std::to_string(impl_->latency_ms) + " protocols=tcp" +
        " ! rtph264depay" + " ! h264parse config-interval=1" +
        " ! video/x-h264,stream-format=byte-stream,alignment=au" +
        " ! appsink name=sink emit-signals=true sync=false drop=true max-buffers=1";

    GError* error = nullptr;
    impl_->pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
    if (!impl_->pipeline || error) {
        std::cerr << "[RtspDemuxer] Pipeline kurulamadı: "
                  << (error ? error->message : "bilinmiyor") << "\n";
        if (error) g_error_free(error);
        return false;
    }

    impl_->appsink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "sink");
    if (!impl_->appsink) {
        std::cerr << "[RtspDemuxer] appsink elementi bulunamadı\n";
        return false;
    }

    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = onNewSample;
    gst_app_sink_set_callbacks(GST_APP_SINK(impl_->appsink), &callbacks, impl_.get(),
                                nullptr);

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(impl_->pipeline));
    impl_->bus_watch_id = gst_bus_add_watch(bus, onBusMessage, impl_.get());
    gst_object_unref(bus);

    impl_->loop = g_main_loop_new(nullptr, FALSE);

    GstStateChangeReturn ret = gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[RtspDemuxer] Pipeline PLAYING durumuna geçemedi — "
                     "RTSP adresini/kimlik bilgilerini kontrol edin\n";
        return false;
    }

    impl_->loop_thread = std::thread([this] { g_main_loop_run(impl_->loop); });

    return true;
}

void RtspDemuxer::stop() {
    if (impl_) {
        impl_->stop();
    }
}

RtspDemuxer::Stats RtspDemuxer::getStats() const {
    return Stats{
        impl_->packets_received.load(std::memory_order_relaxed),
        impl_->bytes_received.load(std::memory_order_relaxed),
        impl_->packets_dropped.load(std::memory_order_relaxed),
    };
}
