/**
 * *****************************************************************************
 * @file      camera_stream_node.hpp
 * @author    Tapbot Development Team & SwiX Team
 * @date      September 2026
 * @brief     Node phát camera đa luồng siêu nhẹ cho Jetson Orin.
 *            Tận dụng hardware VIC (Video Image Compositor) và NVENC (H.265/H.264).
 *            Tích hợp cơ chế Lazy Publishing: Chỉ xử lý & publish khi CÓ SUBSCRIBER.
 *            Tích hợp Birdview Stitching (OpenCV CUDA) & Hardware NVENC Birdview.
 * *****************************************************************************
 */
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <cuda_runtime.h>

#include "camera_stream/face_anonymizer.hpp"
#include "camera_stream/birdview_processor.hpp"
#include "camera_stream/birdview_encoder.hpp"

namespace camera_stream {

struct CameraConfig {
    std::string name;
    std::string devicePath;
    std::string opticalFrame;
    std::string imageTopic;
    std::string compressedTopic;
    int flipMethod = 0;
};

struct CameraStream {
    CameraConfig config;
    int index = 0;

    GstElement* pipeline = nullptr;
    GstElement* rawValve = nullptr;
    GstElement* compValve = nullptr;
    GstElement* rawAppSink = nullptr;
    GstElement* compAppSink = nullptr;
    GstBus* bus = nullptr;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rawPublisher;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressedPublisher;

    sensor_msgs::msg::Image rawMessageTemplate;
    sensor_msgs::msg::CompressedImage compMessageTemplate;

    bool rawValveOpen = true;
    bool compValveOpen = true;
    bool pipelinePlaying = false;
    bool capsNegotiated = false;

    // Cache trạng thái subscriber siêu nhẹ
    std::atomic<bool> hasRawSubscribers{false};
    std::atomic<bool> hasCompSubscribers{false};
    std::atomic<bool> hasAnySubscribers{false};

    struct DmabufMapping {
        int fd = -1;
        cudaExternalMemory_t extMem = nullptr;
        void* devPtr = nullptr;
    };
    std::vector<DmabufMapping> anonDmabufMappings;

    std::atomic<bool> isReady{false};
    std::atomic<int64_t> publishedCount{0};

    std::shared_ptr<FaceAnonymizer> anonymizer;
};

class CameraStreamNode : public rclcpp::Node {
public:
    explicit CameraStreamNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~CameraStreamNode() override;

    bool isBirdviewActive() const;

private:
    void loadParameters();
    void initializeCameras();
    void initializeBirdview();
    bool buildPipeline(CameraStream& camera);
    void startPipelines();
    void stopPipelines();
    void updateValves();
    void processBirdview();

    // Callbacks từ GStreamer appsink
    static GstFlowReturn onNewRawSample(GstAppSink* sink, gpointer userData);
    static GstFlowReturn onNewCompressedSample(GstAppSink* sink, gpointer userData);

    // GStreamer Pad Probe can thiệp trực tiếp vào buffer NVMM VRAM
    static GstPadProbeReturn onNvmmBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer userData);

    void processRawSample(GstSample* sample, CameraStream* camera);
    void processCompressedSample(GstSample* sample, CameraStream* camera);

    rclcpp::Time getCaptureStamp(const CameraStream& camera, GstBuffer* buffer);
    void checkPipelineHealth();

    // Cấu hình chung
    int captureWidth_ = 1920;
    int captureHeight_ = 1536;
    int captureFramerate_ = 30;
    std::string captureFormat_ = "YUY2";

    int outputWidth_ = 640;
    int outputHeight_ = 480;
    int outputFramerate_ = 15;
    std::string outputEncoding_ = "bgra8";
    bool enableRaw_ = true;
    bool enableCompressed_ = true;
    int jpegQuality_ = 85;
    std::string compCodec_ = "h265";
    int compBitrate_ = 180000;

    // Cấu hình Privacy Anonymizer
    int anonymizeHoldFrames_ = 5;
    int anonymizeMosaicSize_ = 5;

    // 1. Nhánh Face
    bool enableFace_ = true;
    std::string faceModelPath_ = "models/face_model/version-RFB-320_fp16.engine";
    float faceConfThresh_ = 0.20f;
    float faceNmsThresh_ = 0.30f;
    float faceBoxScale_ = 1.15f;

    // 2. Nhánh Plate
    bool enablePlate_ = true;
    std::string plateModelPath_ = "models/plate_model/plate_ocr_det_640_fp16.engine";
    float plateConfThresh_ = 0.20f;
    float plateNmsThresh_ = 0.35f;
    float plateBoxScale_ = 1.15f;

    std::shared_ptr<FaceAnonymizer> faceAnonymizer_;

    // Cấu hình Birdview System
    bool enableBirdview_ = true;
    int birdviewFps_ = 15;
    int birdviewBitrate_ = 250000;
    std::string birdviewConfigFile_ = "config/birdview_ground.json";
    std::string birdviewCarImgPath_ = "config/SwiX.png";
    std::string birdviewRawTopic_ = "/camera/birdview/raw";
    std::string birdviewCompressedTopic_ = "/camera/birdview/raw/compress";

    std::unique_ptr<BirdviewProcessor> birdviewProcessor_;
    std::unique_ptr<BirdviewEncoder> birdviewEncoder_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr birdviewRawPublisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr birdviewCompressedPublisher_;
    rclcpp::TimerBase::SharedPtr birdviewTimer_;

    mutable std::mutex birdviewMutex_;
    std::unordered_map<std::string, cv::Mat> birdviewFrames_;
    builtin_interfaces::msg::Time birdviewLastStamp_;

    // ROS 2 Services điều khiển động lúc runtime
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srvEnableFace_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srvEnablePlate_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srvEnableAll_;

    // Dynamic parameter callback handle
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr paramCallbackHandle_;

    std::vector<std::string> cameraNames_;
    std::vector<std::unique_ptr<CameraStream>> cameras_;

    rclcpp::TimerBase::SharedPtr healthTimer_;
    rclcpp::TimerBase::SharedPtr valveTimer_;
};

}  // namespace camera_stream
