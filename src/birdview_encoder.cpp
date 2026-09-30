/**
 * *****************************************************************************
 * @file      birdview_encoder.cpp
 * @author    SwiX Team
 * @date      September 2026
 * @brief     Cài đặt Hardware Video Encoder (NVENC H.265/H.264) cho ảnh Birdview.
 * *****************************************************************************
 */
#include "camera_stream/birdview_encoder.hpp"

#include <iostream>
#include <sstream>

namespace camera_stream {

BirdviewEncoder::BirdviewEncoder() = default;

BirdviewEncoder::~BirdviewEncoder() {
    stop();
}

bool BirdviewEncoder::initialize(int width, int height, int fps, int bitrate, const std::string& codec) {
    stop();

    width_ = width;
    height_ = height;
    fps_ = (fps > 0) ? fps : 15;
    bitrate_ = (bitrate > 0) ? bitrate : 250000;
    codec_ = codec;

    current_pts_ = 0;
    frame_duration_ = gst_util_uint64_scale_int(1, GST_SECOND, fps_);

    std::ostringstream ss;
    ss << "appsrc name=bv_src is-live=true format=time "
       << "caps=\"video/x-raw, format=BGR, width=" << width_
       << ", height=" << height_
       << ", framerate=" << fps_ << "/1\" ! "
       << "videoconvert ! video/x-raw, format=BGRx ! "
       << "nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! ";

    if (codec_ == "h264") {
        ss << "nvv4l2h264enc bitrate=" << bitrate_
           << " insert-sps-pps=true idrinterval=" << (fps_ * 2)
           << " maxperf-enable=1 preset-level=1 ! "
           << "h264parse config-interval=-1 ! video/x-h264, stream-format=byte-stream ! ";
    } else {
        // Mặc định H.265
        ss << "nvv4l2h265enc bitrate=" << bitrate_
           << " insert-sps-pps=true idrinterval=" << (fps_ * 2)
           << " maxperf-enable=1 preset-level=1 ! "
           << "h265parse config-interval=-1 ! video/x-h265, stream-format=byte-stream ! ";
    }

    ss << "appsink name=bv_sink emit-signals=false max-buffers=1 drop=true sync=false";

    const std::string pipeline_str = ss.str();
    GError* err = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &err);
    if (err != nullptr || pipeline_ == nullptr) {
        std::cerr << "[BIRDVIEW_ENCODER] Loi khoi tao pipeline GStreamer: "
                  << (err ? err->message : "Unknown error") << "\n";
        if (err) g_error_free(err);
        pipeline_ = nullptr;
        return false;
    }

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "bv_src");
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "bv_sink");

    if (!appsrc_ || !appsink_) {
        std::cerr << "[BIRDVIEW_ENCODER] Khong the tim thay bv_src hoac bv_sink trong pipeline!\n";
        stop();
        return false;
    }

    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    std::cout << "[BIRDVIEW_ENCODER] Khoi tao pipeline NVENC H.265 Birdview thanh cong ("
              << width_ << "x" << height_ << "@" << fps_ << "fps, " << bitrate_ << " bps)\n";
    return true;
}

bool BirdviewEncoder::encode(const cv::Mat& bgr_frame, std::vector<uint8_t>& out_compressed) {
    if (!pipeline_ || !appsrc_ || !appsink_) {
        return false;
    }

    // Dọn toàn bộ bản tin bus tích lũy để tránh leak bộ nhớ dần dần
    {
        GstBus* bus = gst_element_get_bus(pipeline_);
        if (bus) {
            GstMessage* msg;
            while ((msg = gst_bus_pop(bus)) != nullptr) {
                gst_message_unref(msg);
            }
            gst_object_unref(bus);
        }
    }

    if (bgr_frame.empty() || bgr_frame.cols != width_ || bgr_frame.rows != height_) {
        return false;
    }

    const size_t data_size = bgr_frame.total() * bgr_frame.elemSize();
    GstBuffer* buf = gst_buffer_new_allocate(nullptr, data_size, nullptr);
    if (!buf) return false;

    gst_buffer_fill(buf, 0, bgr_frame.data, data_size);
    GST_BUFFER_PTS(buf) = current_pts_;
    GST_BUFFER_DURATION(buf) = frame_duration_;
    current_pts_ += frame_duration_;

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buf);
    if (ret != GST_FLOW_OK) {
        return false;
    }

    // Lấy sample nén từ appsink (timeout 25ms)
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 25 * GST_MSECOND);
    if (!sample) {
        return false;
    }

    GstBuffer* out_buf = gst_sample_get_buffer(sample);
    if (out_buf) {
        GstMapInfo map;
        if (gst_buffer_map(out_buf, &map, GST_MAP_READ)) {
            out_compressed.assign(map.data, map.data + map.size);
            gst_buffer_unmap(out_buf, &map);
        }
    }

    gst_sample_unref(sample);
    return !out_compressed.empty();
}

void BirdviewEncoder::stop() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (appsrc_) {
            gst_object_unref(appsrc_);
            appsrc_ = nullptr;
        }
        if (appsink_) {
            gst_object_unref(appsink_);
            appsink_ = nullptr;
        }
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

} // namespace camera_stream
