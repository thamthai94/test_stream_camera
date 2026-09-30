/**
 * *****************************************************************************
 * @file      birdview_encoder.hpp
 * @author    SwiX Team
 * @date      September 2026
 * @brief     Hardware Video Encoder (NVENC H.265/H.264) cho ảnh Birdview trên Jetson.
 *            Xuất byte-stream H.265 tối ưu cho RTSP Pass-Through.
 * *****************************************************************************
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <opencv2/core.hpp>

namespace camera_stream {

class BirdviewEncoder {
public:
    BirdviewEncoder();
    ~BirdviewEncoder();

    bool initialize(int width, int height, int fps, int bitrate, const std::string& codec = "h265");
    bool encode(const cv::Mat& bgr_frame, std::vector<uint8_t>& out_compressed);
    void stop();

    bool isInitialized() const { return pipeline_ != nullptr; }

private:
    int width_ = 600;
    int height_ = 600;
    int fps_ = 15;
    int bitrate_ = 250000;
    std::string codec_ = "h265";

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;

    uint64_t current_pts_ = 0;
    uint64_t frame_duration_ = 0;
};

} // namespace camera_stream
