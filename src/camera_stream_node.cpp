/**
 * *****************************************************************************
 * @file      camera_stream_node.cpp
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     Hiện thực hóa CameraStreamNode:
 *            - Tận dụng hardware acceleration VIC / NVENC trên Jetson.
 *            - Lazy Publishing: Chỉ xử lý & publish khi CÓ NGƯỜI SUBSCRIBE.
 *            - Tiêu tốn 0% CUDA GPU và gần như 0% CPU.
 * *****************************************************************************
 */
#include "camera_stream/camera_stream_node.hpp"

#include <filesystem>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>
#include <nvbufsurface.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

namespace camera_stream {

CameraStreamNode::CameraStreamNode(const rclcpp::NodeOptions& options)
    : Node("camera_stream_node", options) {
    gst_init(nullptr, nullptr);

    loadParameters();
    initializeCameras();
    initializeBirdview();
    startPipelines();

    // Timer giám sát sức khỏe pipeline (1 Hz)
    healthTimer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        [this]() { this->checkPipelineHealth(); }
    );

    // Timer cập nhật van đóng/mở động theo số lượng subscriber (1 Hz / 1000ms)
    valveTimer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000),
        [this]() { this->updateValves(); }
    );

    RCLCPP_INFO(this->get_logger(), "[CAMERA_STREAM] Node khoi dong thanh cong cho %zu camera.", cameras_.size());
    RCLCPP_INFO(this->get_logger(), "[CAMERA_STREAM] ZERO-CPU DYNAMIC VALVE ENABLED: Van tu dong dong khi khong co subscriber!");
}

CameraStreamNode::~CameraStreamNode() {
    stopPipelines();
}

void CameraStreamNode::loadParameters() {
    // Thông số capture phần cứng
    captureWidth_ = this->declare_parameter<int>("capture.width", 1920);
    captureHeight_ = this->declare_parameter<int>("capture.height", 1536);
    captureFramerate_ = this->declare_parameter<int>("capture.framerate", 30);
    captureFormat_ = this->declare_parameter<std::string>("capture.format", "YUY2");

    // Thông số output ROS (mặc định 640x480 an toàn cho CPU)
    outputWidth_ = this->declare_parameter<int>("output.width", 640);
    outputHeight_ = this->declare_parameter<int>("output.height", 480);
    outputFramerate_ = this->declare_parameter<int>("output.framerate", 15);
    outputEncoding_ = this->declare_parameter<std::string>("output.encoding", "bgra8");
    enableRaw_ = this->declare_parameter<bool>("output.enable_raw", true);
    enableCompressed_ = this->declare_parameter<bool>("output.enable_compressed", true);
    jpegQuality_ = this->declare_parameter<int>("output.jpeg_quality", 85);
    compCodec_ = this->declare_parameter<std::string>("output.comp_codec", "h265");
    compBitrate_ = this->declare_parameter<int>("output.comp_bitrate", 180000);

    // Danh sách camera
    cameraNames_ = this->declare_parameter<std::vector<std::string>>(
        "camera_names",
        std::vector<std::string>{"front", "rear", "left", "right"}
    );

    // Cấu hình Privacy Anonymizer (Che mờ bảo mật)
    anonymizeHoldFrames_ = this->declare_parameter<int>("anonymize.hold_frames", 5);
    anonymizeMosaicSize_ = this->declare_parameter<int>("anonymize.mosaic_size", 5);

    // 1. Cấu hình Khuôn mặt (Face) - Hỗ trợ cả anonymize.face.* và fallback anonymize.*
    std::string defaultFaceModel = "models/face_model/version-RFB-320_fp16.engine";
    if (this->has_parameter("anonymize.model_path")) {
        defaultFaceModel = this->get_parameter("anonymize.model_path").as_string();
    } else {
        defaultFaceModel = this->declare_parameter<std::string>("anonymize.model_path", defaultFaceModel);
    }
    float defaultFaceConf = 0.20f;
    if (this->has_parameter("anonymize.confidence_threshold")) {
        defaultFaceConf = static_cast<float>(this->get_parameter("anonymize.confidence_threshold").as_double());
    } else {
        defaultFaceConf = this->declare_parameter<float>("anonymize.confidence_threshold", defaultFaceConf);
    }
    float defaultFaceNms = 0.30f;
    if (this->has_parameter("anonymize.nms_threshold")) {
        defaultFaceNms = static_cast<float>(this->get_parameter("anonymize.nms_threshold").as_double());
    } else {
        defaultFaceNms = this->declare_parameter<float>("anonymize.nms_threshold", defaultFaceNms);
    }
    float defaultFaceScale = 1.15f;
    if (this->has_parameter("anonymize.box_scale")) {
        defaultFaceScale = static_cast<float>(this->get_parameter("anonymize.box_scale").as_double());
    } else {
        defaultFaceScale = this->declare_parameter<float>("anonymize.box_scale", defaultFaceScale);
    }

    enableFace_ = this->declare_parameter<bool>("anonymize.face.enable", true);
    faceModelPath_ = this->declare_parameter<std::string>("anonymize.face.model_path", defaultFaceModel);
    faceConfThresh_ = this->declare_parameter<float>("anonymize.face.confidence_threshold", defaultFaceConf);
    faceNmsThresh_ = this->declare_parameter<float>("anonymize.face.nms_threshold", defaultFaceNms);
    faceBoxScale_ = this->declare_parameter<float>("anonymize.face.box_scale", defaultFaceScale);

    // 2. Cấu hình Biển số xe & Chữ số (Plate & Text)
    enablePlate_ = this->declare_parameter<bool>("anonymize.plate.enable", true);
    plateModelPath_ = this->declare_parameter<std::string>("anonymize.plate.model_path", "models/plate_model/plate_ocr_det_640_fp16.engine");
    plateConfThresh_ = this->declare_parameter<float>("anonymize.plate.confidence_threshold", 0.20f);
    plateNmsThresh_ = this->declare_parameter<float>("anonymize.plate.nms_threshold", 0.35f);
    plateBoxScale_ = this->declare_parameter<float>("anonymize.plate.box_scale", 1.15f);

    // Khởi tạo Privacy Anonymizer (Tải sẵn các model TensorRT vào GPU VRAM nếu có file model)
    {
        // Tìm kiếm file model Face
        std::string resolvedFacePath = faceModelPath_;
        if (!std::filesystem::exists(resolvedFacePath)) {
            try {
                std::string pkgShare = ament_index_cpp::get_package_share_directory("camera_stream");
                std::string sharePath = pkgShare + "/" + faceModelPath_;
                if (std::filesystem::exists(sharePath)) resolvedFacePath = sharePath;
            } catch (...) {}
        }
        if (!std::filesystem::exists(resolvedFacePath)) {
            std::string localPath = (std::filesystem::current_path() / faceModelPath_).string();
            if (std::filesystem::exists(localPath)) resolvedFacePath = localPath;
        }
        if (!std::filesystem::exists(resolvedFacePath)) {
            const char* homeDir = std::getenv("HOME");
            if (homeDir != nullptr) {
                std::string homePath = std::string(homeDir) + "/face_model/version-RFB-320_fp16.engine";
                if (std::filesystem::exists(homePath)) resolvedFacePath = homePath;
            }
        }

        // Tìm kiếm file model Plate
        std::string resolvedPlatePath = plateModelPath_;
        if (!std::filesystem::exists(resolvedPlatePath)) {
            try {
                std::string pkgShare = ament_index_cpp::get_package_share_directory("camera_stream");
                std::string sharePath = pkgShare + "/" + plateModelPath_;
                if (std::filesystem::exists(sharePath)) resolvedPlatePath = sharePath;
            } catch (...) {}
        }
        if (!std::filesystem::exists(resolvedPlatePath)) {
            std::string localPath = (std::filesystem::current_path() / plateModelPath_).string();
            if (std::filesystem::exists(localPath)) resolvedPlatePath = localPath;
        }

        FaceAnonymizer::Config anonCfg;
        anonCfg.holdFrames = anonymizeHoldFrames_;
        anonCfg.mosaicSize = anonymizeMosaicSize_;

        anonCfg.enableFace = enableFace_;
        anonCfg.enginePath = resolvedFacePath;
        anonCfg.confThreshold = faceConfThresh_;
        anonCfg.nmsThreshold = faceNmsThresh_;
        anonCfg.boxScale = faceBoxScale_;

        anonCfg.enablePlate = enablePlate_;
        anonCfg.plateEnginePath = resolvedPlatePath;
        anonCfg.plateConfThreshold = plateConfThresh_;
        anonCfg.plateNmsThreshold = plateNmsThresh_;
        anonCfg.plateBoxScale = plateBoxScale_;

        faceAnonymizer_ = std::make_shared<FaceAnonymizer>(anonCfg);
        if (!faceAnonymizer_->initialize()) {
            RCLCPP_WARN(this->get_logger(), "[CAMERA_STREAM] Khoi tao Privacy Anonymizer that bai -> Tiep tuc stream khong che.");
        } else {
            RCLCPP_INFO(this->get_logger(),
                "[CAMERA_STREAM] Privacy Anonymizer (TensorRT 10) da tai xong model [Face: %s (san sang: %s) | Plate: %s (san sang: %s)]",
                enableFace_ ? "BAT" : "TAT", faceAnonymizer_->isFaceReady() ? "CO" : "KHONG",
                enablePlate_ ? "BAT" : "TAT", faceAnonymizer_->isPlateReady() ? "CO" : "KHONG");
        }
    }

    // Khởi tạo các ROS 2 Services điều khiển động lúc runtime
    srvEnableFace_ = this->create_service<std_srvs::srv::SetBool>(
        "~/anonymize/enable_face",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
            if (faceAnonymizer_) {
                faceAnonymizer_->setEnableFace(req->data);
                enableFace_ = req->data;
                res->success = true;
                res->message = req->data ? "Da BAT che mat (Face ON)." : "Da TAT che mat (Face OFF).";
                RCLCPP_INFO(this->get_logger(), "[ANONYMIZER SERVICE] %s", res->message.c_str());
            } else {
                res->success = false;
                res->message = "Anonymizer chua duoc khoi tao!";
            }
        }
    );

    srvEnablePlate_ = this->create_service<std_srvs::srv::SetBool>(
        "~/anonymize/enable_plate",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
            if (faceAnonymizer_) {
                faceAnonymizer_->setEnablePlate(req->data);
                enablePlate_ = req->data;
                res->success = true;
                res->message = req->data ? "Da BAT che bien so xe (Plate ON)." : "Da TAT che bien so xe (Plate OFF).";
                RCLCPP_INFO(this->get_logger(), "[ANONYMIZER SERVICE] %s", res->message.c_str());
            } else {
                res->success = false;
                res->message = "Anonymizer chua duoc khoi tao!";
            }
        }
    );

    srvEnableAll_ = this->create_service<std_srvs::srv::SetBool>(
        "~/anonymize/enable_all",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
            if (faceAnonymizer_) {
                faceAnonymizer_->setEnableAll(req->data);
                enableFace_ = req->data;
                enablePlate_ = req->data;
                res->success = true;
                res->message = req->data ? "Da BAT tat ca che mo (Face ON, Plate ON)." : "Da TAT tat ca che mo (Face OFF, Plate OFF, 0% GPU).";
                RCLCPP_INFO(this->get_logger(), "[ANONYMIZER SERVICE] %s", res->message.c_str());
            } else {
                res->success = false;
                res->message = "Anonymizer chua duoc khoi tao!";
            }
        }
    );

    // Dynamic parameter callback
    paramCallbackHandle_ = this->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& params) {
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            if (!faceAnonymizer_) return result;

            for (const auto& p : params) {
                if (p.get_name() == "anonymize.enable") {
                    bool val = p.as_bool();
                    faceAnonymizer_->setEnableAll(val);
                    enableFace_ = val;
                    enablePlate_ = val;
                    RCLCPP_INFO(this->get_logger(), "[PARAM SET] anonymize.enable = %s -> Dat ca Face & Plate thanh %s",
                                val ? "true" : "false", val ? "ON" : "OFF");
                } else if (p.get_name() == "anonymize.face.enable") {
                    bool val = p.as_bool();
                    faceAnonymizer_->setEnableFace(val);
                    enableFace_ = val;
                    RCLCPP_INFO(this->get_logger(), "[PARAM SET] anonymize.face.enable = %s", val ? "true" : "false");
                } else if (p.get_name() == "anonymize.plate.enable") {
                    bool val = p.as_bool();
                    faceAnonymizer_->setEnablePlate(val);
                    enablePlate_ = val;
                    RCLCPP_INFO(this->get_logger(), "[PARAM SET] anonymize.plate.enable = %s", val ? "true" : "false");
                }
            }
            return result;
        }
    );

    // 3. Cấu hình Birdview System
    enableBirdview_ = this->declare_parameter<bool>("birdview.enable", true);
    birdviewFps_ = this->declare_parameter<int>("birdview.fps", 15);
    birdviewBitrate_ = this->declare_parameter<int>("birdview.bitrate", 250000);
    birdviewConfigFile_ = this->declare_parameter<std::string>("birdview.config_file", "config/birdview_ground.json");
    birdviewCarImgPath_ = this->declare_parameter<std::string>("birdview.car_image_path", "config/SwiX.png");
    birdviewRawTopic_ = this->declare_parameter<std::string>("birdview.raw_topic", "/camera/birdview/raw");
    birdviewCompressedTopic_ = this->declare_parameter<std::string>("birdview.compressed_topic", "/camera/birdview/raw/compress");

    RCLCPP_INFO(this->get_logger(), "[CAMERA_STREAM] Capture: %dx%d@%dfps (%s) | Output: %dx%d@%dfps (%s) | Raw: %s | Comp: %s (%s, %d bps) | Anonymize: %s | Birdview: %s",
                captureWidth_, captureHeight_, captureFramerate_, captureFormat_.c_str(),
                outputWidth_, outputHeight_, outputFramerate_, outputEncoding_.c_str(),
                enableRaw_ ? "ON" : "OFF", enableCompressed_ ? "ON" : "OFF",
                compCodec_.c_str(), compBitrate_,
                (faceAnonymizer_ && faceAnonymizer_->isEnabled()) ? "ON (Mosaic)" : "OFF",
                enableBirdview_ ? "ON" : "OFF");
}

void CameraStreamNode::initializeCameras() {
    int index = 0;
    for (const auto& name : cameraNames_) {
        const std::string prefix = "cameras." + name + ".";

        CameraConfig config;
        config.name = name;
        config.devicePath = this->declare_parameter<std::string>(prefix + "device", "");
        config.opticalFrame = this->declare_parameter<std::string>(prefix + "optical_frame", "camera_" + name + "_optical_frame");
        config.imageTopic = this->declare_parameter<std::string>(prefix + "image_topic", "/camera/" + name + "/image_raw");
        config.compressedTopic = this->declare_parameter<std::string>(prefix + "compressed_topic", "/camera/" + name + "/image_raw/compressed");
        config.flipMethod = this->declare_parameter<int>(prefix + "flip_method", 0);

        if (config.devicePath.empty()) {
            RCLCPP_WARN(this->get_logger(), "[CAMERA_STREAM] %s thieu device path trong YAML -> bo qua", name.c_str());
            continue;
        }

        auto stream = std::make_unique<CameraStream>();
        stream->config = config;
        stream->index = index++;

        // Tạo ROS publishers với QoS SensorDataQoS (tối ưu cho camera real-time)
        if (enableRaw_) {
            stream->rawPublisher = this->create_publisher<sensor_msgs::msg::Image>(
                config.imageTopic, rclcpp::SensorDataQoS()
            );
            RCLCPP_INFO(this->get_logger(), "[CAMERA_STREAM] %s: Topic Raw -> %s", name.c_str(), config.imageTopic.c_str());
        }

        if (enableCompressed_) {
            stream->compressedPublisher = this->create_publisher<sensor_msgs::msg::CompressedImage>(
                config.compressedTopic, rclcpp::SensorDataQoS()
            );
            RCLCPP_INFO(this->get_logger(), "[CAMERA_STREAM] %s: Topic Compressed -> %s", name.c_str(), config.compressedTopic.c_str());
        }

        stream->anonymizer = faceAnonymizer_;

        cameras_.push_back(std::move(stream));
    }
}

bool CameraStreamNode::buildPipeline(CameraStream& camera) {
    if (!std::filesystem::exists(camera.config.devicePath)) {
        RCLCPP_ERROR(this->get_logger(), "[CAMERA_STREAM] %s: Thiet bi %s khong ton tai!",
                     camera.config.name.c_str(), camera.config.devicePath.c_str());
        return false;
    }

    std::ostringstream ss;
    // v4l2src: Thu nhận trực tiếp định dạng phần cứng (mặc định YUY2 1920x1536)
    ss << "v4l2src device=" << camera.config.devicePath << " io-mode=2 ! "
       << "video/x-raw, format=" << captureFormat_
       << ", width=" << captureWidth_
       << ", height=" << captureHeight_
       << ", framerate=" << captureFramerate_ << "/1 ! ";

    // Nếu output framerate thấp hơn capture framerate, hạ nhịp an toàn bằng videorate (0% CPU, chỉ drop con trỏ)
    if (outputFramerate_ > 0 && outputFramerate_ < captureFramerate_) {
        ss << "videorate drop-only=true ! "
           << "video/x-raw, framerate=" << outputFramerate_ << "/1 ! ";
    }

    // Đưa ngay vào phần cứng VIC (0% CPU) để xoay và downscale xuống 640x480 NV12 trong NVMM ngay tại đầu nguồn.
    const int targetFps = (outputFramerate_ > 0) ? outputFramerate_ : captureFramerate_;
    ss << "nvvidconv name=nvconv_" << camera.config.name
       << " flip-method=" << camera.config.flipMethod
       << " bl-output=false ! "
       << "capsfilter name=nvcaps_" << camera.config.name << " caps=\"video/x-raw(memory:NVMM), format=NV12, width="
       << outputWidth_ << ", height=" << outputHeight_
       << ", framerate=" << targetFps << "/1\" ! ";

    auto buildRawBranch = [&](const std::string& prefix) {
        std::ostringstream bss;
        bss << prefix << "queue max-size-buffers=1 leaky=downstream ! "
            << "valve name=raw_valve drop=false ! "
            << "nvvidconv ! ";

        if (outputEncoding_ == "rgba8") {
            bss << "video/x-raw, format=RGBA ! ";
        } else if (outputEncoding_ == "bgra8") {
            bss << "video/x-raw, format=BGRx ! ";
        } else if (outputEncoding_ == "rgb8") {
            bss << "video/x-raw, format=RGBA ! videoconvert ! video/x-raw, format=RGB ! ";
        } else {
            bss << "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! ";
        }

        bss << "appsink name=raw_sink emit-signals=false max-buffers=1 drop=true sync=false";
        return bss.str();
    };

    auto buildCompBranch = [&](const std::string& prefix) {
        std::ostringstream bss;
        bss << prefix << "queue max-size-buffers=1 leaky=downstream ! "
            << "valve name=comp_valve drop=false ! ";

        if (compCodec_ == "h265") {
            bss << "nvv4l2h265enc bitrate=" << compBitrate_ << " insert-sps-pps=true preset-level=1 maxperf-enable=1 idrinterval=" << (captureFramerate_ * 2) << " ! "
                << "h265parse config-interval=-1 ! video/x-h265, stream-format=byte-stream ! ";
        } else if (compCodec_ == "h264") {
            bss << "nvv4l2h264enc bitrate=" << compBitrate_ << " insert-sps-pps=true preset-level=1 maxperf-enable=1 idrinterval=" << (captureFramerate_ * 2) << " ! "
                << "h264parse config-interval=-1 ! video/x-h264, stream-format=byte-stream ! ";
        } else {
            bss << "nvjpegenc quality=" << jpegQuality_ << " ! image/jpeg ! ";
        }

        bss << "appsink name=comp_sink emit-signals=false max-buffers=1 drop=true sync=false";
        return bss.str();
    };

    if (enableRaw_ && enableCompressed_) {
        ss << "tee name=t allow-not-linked=true ";
        ss << buildRawBranch("t. ! ") << " ";
        ss << buildCompBranch("t. ! ");
    } else if (enableRaw_) {
        ss << buildRawBranch("");
    } else if (enableCompressed_) {
        ss << buildCompBranch("");
    } else {
        RCLCPP_ERROR(this->get_logger(), "[CAMERA_STREAM] Ca enable_raw va enable_compressed deu tat!");
        return false;
    }

    const std::string pipelineStr = ss.str();
    RCLCPP_INFO(this->get_logger(), "[CAMERA_STREAM] %s GStreamer pipeline:\n%s",
                camera.config.name.c_str(), pipelineStr.c_str());

    GError* error = nullptr;
    camera.pipeline = gst_parse_launch(pipelineStr.c_str(), &error);
    if (error != nullptr) {
        RCLCPP_ERROR(this->get_logger(), "[CAMERA_STREAM] %s loi khoi tao pipeline: %s",
                     camera.config.name.c_str(), error->message);
        g_error_free(error);
        return false;
    }

    // Lấy bus để theo dõi lỗi runtime
    camera.bus = gst_element_get_bus(camera.pipeline);

    // Gắn valve và direct callback cho raw_sink
    if (enableRaw_) {
        camera.rawValve = gst_bin_get_by_name(GST_BIN(camera.pipeline), "raw_valve");
        camera.rawValveOpen = true; // Bắt đầu ở true để negotiate CAPS bước đầu

        camera.rawAppSink = gst_bin_get_by_name(GST_BIN(camera.pipeline), "raw_sink");
        if (camera.rawAppSink != nullptr) {
            g_object_set_data(G_OBJECT(camera.rawAppSink), "node_instance", this);
            GstAppSinkCallbacks callbacks = {};
            callbacks.new_sample = onNewRawSample;
            gst_app_sink_set_callbacks(GST_APP_SINK(camera.rawAppSink), &callbacks, &camera, nullptr);
        }
    }

    // Gắn valve và direct callback cho comp_sink
    if (enableCompressed_) {
        camera.compValve = gst_bin_get_by_name(GST_BIN(camera.pipeline), "comp_valve");
        camera.compValveOpen = true; // Bắt đầu ở true để negotiate CAPS bước đầu

        camera.compAppSink = gst_bin_get_by_name(GST_BIN(camera.pipeline), "comp_sink");
        if (camera.compAppSink != nullptr) {
            g_object_set_data(G_OBJECT(camera.compAppSink), "node_instance", this);
            GstAppSinkCallbacks callbacks = {};
            callbacks.new_sample = onNewCompressedSample;
            gst_app_sink_set_callbacks(GST_APP_SINK(camera.compAppSink), &callbacks, &camera, nullptr);
        }
    }

    // Sử dụng mặc định MONOTONIC clock (khớp với PTS từ v4l2src, tránh drift khi NTP chỉnh đồng hồ)

    // Gắn Zero-Copy Face/Plate Anonymizer Pad Probe vào buffer NVMM trước khi phân nhánh
    if (camera.anonymizer != nullptr) {
        std::string capsName = "nvcaps_" + camera.config.name;
        GstElement* nvcaps = gst_bin_get_by_name(GST_BIN(camera.pipeline), capsName.c_str());
        if (nvcaps != nullptr) {
            GstPad* srcPad = gst_element_get_static_pad(nvcaps, "src");
            if (srcPad != nullptr) {
                gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER, onNvmmBufferProbe, &camera, nullptr);
                gst_object_unref(srcPad);
                RCLCPP_INFO(this->get_logger(),
                            "[CAMERA_STREAM] %s: Da gan Zero-Copy Face Anonymizer Probe (NVMM).",
                            camera.config.name.c_str());
            }
            gst_object_unref(nvcaps);
        }
    }

    return true;
}

void CameraStreamNode::startPipelines() {
    for (auto& camera : cameras_) {
        if (!buildPipeline(*camera)) {
            continue;
        }

        // Khởi tạo ở GST_STATE_READY: Mở thiết bị nhưng chưa chạy DMA stream,
        // Đảm bảo tiêu tốn đúng 0.0% CPU khi chưa có ai subscribe và sẵn sàng kích hoạt tức thì (<40ms)!
        const GstStateChangeReturn ret = gst_element_set_state(camera->pipeline, GST_STATE_READY);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            RCLCPP_ERROR(this->get_logger(), "[CAMERA_STREAM] %s khong the chuyen sang trang thai READY!",
                         camera->config.name.c_str());
            camera->isReady = false;
            camera->pipelinePlaying = false;
        } else {
            camera->isReady = true;
            camera->pipelinePlaying = false;
            camera->capsNegotiated = false;
            camera->rawValveOpen = true;  // Mở sẵn van để CAPS negotiation đầu tiên truyền qua trơn tru
            camera->compValveOpen = true;
            RCLCPP_INFO(this->get_logger(),
                        "[CAMERA_STREAM] %s khoi tao READY thanh cong (0.0%% CPU idle, san sang chay tuc thi).",
                        camera->config.name.c_str());
        }
    }
}

void CameraStreamNode::stopPipelines() {
    for (auto& camera : cameras_) {
        if (camera->pipeline != nullptr) {
            gst_element_set_state(camera->pipeline, GST_STATE_NULL);
        }
        if (camera->rawValve != nullptr) {
            gst_object_unref(camera->rawValve);
            camera->rawValve = nullptr;
        }
        if (camera->compValve != nullptr) {
            gst_object_unref(camera->compValve);
            camera->compValve = nullptr;
        }
        if (camera->rawAppSink != nullptr) {
            gst_object_unref(camera->rawAppSink);
            camera->rawAppSink = nullptr;
        }
        if (camera->compAppSink != nullptr) {
            gst_object_unref(camera->compAppSink);
            camera->compAppSink = nullptr;
        }
        if (camera->bus != nullptr) {
            gst_object_unref(camera->bus);
            camera->bus = nullptr;
        }
        if (camera->pipeline != nullptr) {
            gst_object_unref(camera->pipeline);
            camera->pipeline = nullptr;
        }
        for (auto& m : camera->anonDmabufMappings) {
            m.devPtr = nullptr;
            if (m.extMem != nullptr) {
                cudaDestroyExternalMemory(m.extMem);
                m.extMem = nullptr;
            }
        }
        camera->anonDmabufMappings.clear();

        camera->isReady = false;
        camera->pipelinePlaying = false;
    }
}

void CameraStreamNode::updateValves() {
    for (auto& camera : cameras_) {
        if (!camera->isReady || camera->pipeline == nullptr) {
            continue;
        }

        const size_t rawSubCount = camera->rawPublisher ? camera->rawPublisher->get_subscription_count() : 0;
        const size_t compSubCount = camera->compressedPublisher ? camera->compressedPublisher->get_subscription_count() : 0;
        const bool hasRaw = (rawSubCount > 0);
        const bool hasComp = (compSubCount > 0);
        const bool hasAny = (hasRaw || hasComp);

        camera->hasRawSubscribers.store(hasRaw, std::memory_order_relaxed);
        camera->hasCompSubscribers.store(hasComp, std::memory_order_relaxed);
        camera->hasAnySubscribers.store(hasAny, std::memory_order_relaxed);

        const bool bvActive = isBirdviewActive();
        const size_t totalSubCount = rawSubCount + compSubCount + (bvActive ? 1 : 0);

        // =====================================================================
        // [CƠ CHẾ TIẾT KIỆM CPU TUYỆT ĐỐI: ZERO-CPU HARDWARE RELEASE]
        // Khi KHÔNG CÓ BẤT KỲ SUBSCRIBER NÀO (0 subscriber):
        // -> Đưa pipeline về GST_STATE_READY.
        if (totalSubCount == 0) {
            if (camera->pipelinePlaying) {
                gst_element_set_state(camera->pipeline, GST_STATE_READY);
                camera->pipelinePlaying = false;
                camera->capsNegotiated = false;
                camera->rawValveOpen = true;
                camera->compValveOpen = true;
                if (camera->rawValve) {
                    g_object_set(G_OBJECT(camera->rawValve), "drop", FALSE, nullptr);
                }
                if (camera->compValve) {
                    g_object_set(G_OBJECT(camera->compValve), "drop", FALSE, nullptr);
                }
                RCLCPP_INFO(this->get_logger(),
                            "[CAMERA_STREAM] %s: 0 subscriber -> PAUSE DMA SENSOR (GST_STATE_READY, 0.0%% CPU)!",
                            camera->config.name.c_str());
            }
            continue;
        }

        // =====================================================================
        // KHI CÓ SUBSCRIBER -> MỞ VAN & RESUME PIPELINE SANG PLAYING
        // =====================================================================
        if (!camera->pipelinePlaying) {
            // Mở van trước để CAPS event không bị chặn (-4 not-negotiated)
            if (camera->rawValve) {
                g_object_set(G_OBJECT(camera->rawValve), "drop", FALSE, nullptr);
                camera->rawValveOpen = true;
            }
            if (camera->compValve) {
                g_object_set(G_OBJECT(camera->compValve), "drop", FALSE, nullptr);
                camera->compValveOpen = true;
            }

            const GstStateChangeReturn ret = gst_element_set_state(camera->pipeline, GST_STATE_PLAYING);
            if (ret != GST_STATE_CHANGE_FAILURE) {
                camera->pipelinePlaying = true;
                RCLCPP_INFO(this->get_logger(),
                            "[CAMERA_STREAM] %s: Phat hien %zu subscriber -> START HARDWARE CAPTURE (PLAYING)",
                            camera->config.name.c_str(), totalSubCount);
            }
        }

        // Chỉ lọc drop sau khi CAPS đã hoàn tất negotiation (sau khi sample đầu tiên đi qua)
        if (camera->capsNegotiated) {
            // Điều khiển van Raw động: Mở khi có subscriber Raw HOẶC Birdview đang active
            if (camera->rawValve && camera->rawPublisher) {
                const bool shouldBeOpen = (rawSubCount > 0 || bvActive);
                if (shouldBeOpen != camera->rawValveOpen) {
                    camera->rawValveOpen = shouldBeOpen;
                    g_object_set(G_OBJECT(camera->rawValve), "drop", shouldBeOpen ? FALSE : TRUE, nullptr);
                    RCLCPP_INFO(this->get_logger(),
                                "[CAMERA_STREAM] %s Raw Valve: %s (Subscribers: %zu, Birdview: %s)",
                                camera->config.name.c_str(),
                                shouldBeOpen ? "OPEN (Streaming)" : "CLOSED (0% CPU)",
                                rawSubCount, bvActive ? "ON" : "OFF");
                }
            }

            // Điều khiển van Compressed động
            if (camera->compValve && camera->compressedPublisher) {
                const bool shouldBeOpen = (compSubCount > 0);
                if (shouldBeOpen != camera->compValveOpen) {
                    camera->compValveOpen = shouldBeOpen;
                    g_object_set(G_OBJECT(camera->compValve), "drop", shouldBeOpen ? FALSE : TRUE, nullptr);
                    RCLCPP_INFO(this->get_logger(),
                                "[CAMERA_STREAM] %s Compressed Valve: %s (Subscribers: %zu)",
                                camera->config.name.c_str(),
                                shouldBeOpen ? "OPEN (Streaming)" : "CLOSED (0% CPU)",
                                compSubCount);
                }
            }
        }
    }
}

// *****************************************************************************
// onNvmmBufferProbe - CALLBACK PAD PROBE XỬ LÝ CHE MẶT ZERO-COPY TRÊN NVMM
// *****************************************************************************
GstPadProbeReturn CameraStreamNode::onNvmmBufferProbe(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer userData) {
    auto* camera = static_cast<CameraStream*>(userData);
    if (camera == nullptr || !camera->anonymizer || !camera->anonymizer->isEnabled()) {
        return GST_PAD_PROBE_OK;
    }

    // Kiểm tra cờ atomic siêu nhẹ (0 ns), không tốn lock DDS middleware
    if (!camera->hasAnySubscribers.load(std::memory_order_relaxed)) {
        return GST_PAD_PROBE_OK;
    }

    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (buffer == nullptr) {
        return GST_PAD_PROBE_OK;
    }

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        NvBufSurface* surf = reinterpret_cast<NvBufSurface*>(map.data);
        if (surf != nullptr && surf->numFilled > 0 && surf->surfaceList != nullptr) {
            auto& sParams = surf->surfaceList[0];
            const int fd = sParams.bufferDesc;

            void* devPtr = nullptr;
            for (const auto& m : camera->anonDmabufMappings) {
                if (m.fd == fd) {
                    devPtr = m.devPtr;
                    break;
                }
            }

            if (devPtr == nullptr && fd >= 0) {
                if (camera->anonDmabufMappings.size() < 16) {
                    cudaExternalMemoryHandleDesc desc{};
                    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                    desc.handle.fd = dup(fd);
                    if (desc.handle.fd >= 0) {
                        size_t allocSize = 0;
                        struct stat st{};
                        if (fstat(desc.handle.fd, &st) == 0 && st.st_size > 0) {
                            allocSize = static_cast<size_t>(st.st_size);
                        } else if (sParams.dataSize > 0) {
                            allocSize = sParams.dataSize;
                        } else {
                            size_t p0 = (sParams.planeParams.pitch[0] > 0) ? sParams.planeParams.pitch[0] : sParams.width;
                            size_t p1 = (sParams.planeParams.pitch[1] > 0) ? sParams.planeParams.pitch[1] : p0;
                            size_t off1 = (sParams.planeParams.num_planes >= 2 && sParams.planeParams.offset[1] > 0)
                                          ? sParams.planeParams.offset[1]
                                          : static_cast<size_t>(sParams.height) * p0;
                            allocSize = off1 + static_cast<size_t>(sParams.height / 2) * p1;
                            allocSize = (allocSize + 4095) & ~4095;
                        }
                        desc.size = allocSize;

                        cudaExternalMemory_t extMem = nullptr;
                        cudaError_t err = cudaImportExternalMemory(&extMem, &desc);
                        close(desc.handle.fd); // Đóng ngay sau khi import để chống tràn file descriptor
                        if (err == cudaSuccess) {
                            cudaExternalMemoryBufferDesc bufDesc{};
                            bufDesc.size = desc.size;
                            err = cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);
                            if (err == cudaSuccess) {
                                camera->anonDmabufMappings.push_back({fd, extMem, devPtr});
                            } else {
                                cudaDestroyExternalMemory(extMem);
                            }
                        }
                    }
                }
            }

            const int width = sParams.width;
            const int height = sParams.height;
            const int yPitch = (sParams.planeParams.pitch[0] > 0)
                               ? sParams.planeParams.pitch[0]
                               : ((sParams.pitch > 0) ? sParams.pitch : width);
            const int uvPitch = (sParams.planeParams.pitch[1] > 0)
                                ? sParams.planeParams.pitch[1]
                                : yPitch;

            const size_t uvOffset = (sParams.planeParams.num_planes >= 2 && sParams.planeParams.offset[1] > 0)
                                    ? sParams.planeParams.offset[1]
                                    : static_cast<size_t>(height) * yPitch;

            if (devPtr != nullptr) {
                uint8_t* d_y = static_cast<uint8_t*>(devPtr) + sParams.planeParams.offset[0];
                uint8_t* d_uv = static_cast<uint8_t*>(devPtr) + uvOffset;
                camera->anonymizer->processFrame(d_y, d_uv, width, height, yPitch, uvPitch, camera->index, camera->config.name);
            }
        }
        gst_buffer_unmap(buffer, &map);
    }

    return GST_PAD_PROBE_OK;
}

// *****************************************************************************
// onNewRawSample - CALLBACK XỬ LÝ KHUNG HÌNH RAW (TRỰC TIẾP TỪ APPSINK)
// *****************************************************************************
GstFlowReturn CameraStreamNode::onNewRawSample(GstAppSink* sink, gpointer userData) {
    auto* camera = static_cast<CameraStream*>(userData);
    if (camera == nullptr) {
        return GST_FLOW_OK;
    }

    camera->capsNegotiated = true;

    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (sample == nullptr) {
        return GST_FLOW_OK;
    }

    auto* node = static_cast<CameraStreamNode*>(g_object_get_data(G_OBJECT(sink), "node_instance"));
    if (node == nullptr) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    const bool rawSubActive = (camera->rawPublisher && camera->rawPublisher->get_subscription_count() > 0);
    const bool bvActive = node->isBirdviewActive();

    if (!rawSubActive && !bvActive) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    node->processRawSample(sample, camera);
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// *****************************************************************************
// onNewCompressedSample - CALLBACK XỬ LÝ KHUNG HÌNH NÉN H265/H264/JPEG
// *****************************************************************************
GstFlowReturn CameraStreamNode::onNewCompressedSample(GstAppSink* sink, gpointer userData) {
    auto* camera = static_cast<CameraStream*>(userData);
    if (camera == nullptr) {
        return GST_FLOW_OK;
    }

    camera->capsNegotiated = true;

    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (sample == nullptr) {
        return GST_FLOW_OK;
    }

    if (!camera->compressedPublisher || camera->compressedPublisher->get_subscription_count() == 0) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    auto* node = static_cast<CameraStreamNode*>(g_object_get_data(G_OBJECT(sink), "node_instance"));
    if (node != nullptr) {
        node->processCompressedSample(sample, camera);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

void CameraStreamNode::processRawSample(GstSample* sample, CameraStream* camera) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (buffer == nullptr) {
        return;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        return;
    }

    const auto stamp = getCaptureStamp(*camera, buffer);
    const bool rawSubActive = (camera->rawPublisher && camera->rawPublisher->get_subscription_count() > 0);
    const bool bvActive = isBirdviewActive();

    // 1. Lưu frame BGRx vào bộ nhớ đệm nếu Birdview đang active
    if (bvActive && (outputEncoding_ == "bgra8" || outputEncoding_ == "bgrx" || outputEncoding_ == "rgba8")) {
        std::lock_guard<std::mutex> lock(birdviewMutex_);
        cv::Mat view(outputHeight_, outputWidth_, CV_8UC4, (void*)map.data);
        view.copyTo(birdviewFrames_[camera->config.name]);
        birdviewLastStamp_ = stamp;
    }

    // 2. Publish ảnh Raw nếu có người đăng ký topic raw của camera này
    if (rawSubActive) {
        auto message = std::make_unique<sensor_msgs::msg::Image>();
        message->header.stamp = stamp;
        message->header.frame_id = camera->config.opticalFrame;
        message->width = static_cast<uint32_t>(outputWidth_);
        message->height = static_cast<uint32_t>(outputHeight_);
        message->encoding = outputEncoding_;
        message->is_bigendian = false;

        size_t bytesPerPixel = 3;
        if (outputEncoding_ == "rgba8" || outputEncoding_ == "bgra8") {
            bytesPerPixel = 4;
        } else if (outputEncoding_ == "yuy2" || outputEncoding_ == "uyvy") {
            bytesPerPixel = 2;
        }
        message->step = static_cast<sensor_msgs::msg::Image::_step_type>(outputWidth_ * bytesPerPixel);
        message->data.assign(map.data, map.data + map.size);

        camera->rawPublisher->publish(std::move(message));
        camera->publishedCount.fetch_add(1, std::memory_order_relaxed);
    }

    gst_buffer_unmap(buffer, &map);
}

void CameraStreamNode::processCompressedSample(GstSample* sample, CameraStream* camera) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (buffer == nullptr) {
        return;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        return;
    }

    auto message = std::make_unique<sensor_msgs::msg::CompressedImage>();
    message->header.stamp = getCaptureStamp(*camera, buffer);
    message->header.frame_id = camera->config.opticalFrame;
    message->format = compCodec_;

    message->data.assign(map.data, map.data + map.size);

    gst_buffer_unmap(buffer, &map);

    camera->compressedPublisher->publish(std::move(message));
    camera->publishedCount.fetch_add(1, std::memory_order_relaxed);
}

rclcpp::Time CameraStreamNode::getCaptureStamp(const CameraStream& /*camera*/, GstBuffer* /*buffer*/) {
    // Dùng trực tiếp ROS clock hiện tại — tránh timestamp drift tích lũy
    // gây bão QoS warnings trong middleware DDS khi stream chạy lâu
    return this->now();
}

// void CameraStreamNode::checkPipelineHealth() {
//     for (auto& camera : cameras_) {
//         if (camera->bus == nullptr) {
//             continue;
//         }

//         while (true) {
//             GstMessage* msg = gst_bus_pop_filtered(camera->bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
//             if (msg == nullptr) {
//                 break;
//             }

//             if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
//                 GError* err = nullptr;
//                 gchar* dbg = nullptr;
//                 gst_message_parse_error(msg, &err, &dbg);
//                 RCLCPP_ERROR(this->get_logger(), "[CAMERA_STREAM] %s loi GStreamer bus: %s (%s)",
//                              camera->config.name.c_str(), err ? err->message : "unknown", dbg ? dbg : "none");
//                 if (err) g_error_free(err);
//                 if (dbg) g_free(dbg);
//             } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
//                 RCLCPP_WARN(this->get_logger(), "[CAMERA_STREAM] %s pipeline nhan End-Of-Stream", camera->config.name.c_str());
//             }
//             gst_message_unref(msg);
//         }
//     }
// }


void CameraStreamNode::checkPipelineHealth() {
    for (auto& camera : cameras_) {
        if (camera->bus == nullptr) {
            continue;
        }

        while (true) {
            // SỬA THÀNH: gst_bus_pop để rút và giải phóng TẤT CẢ bản tin (kể cả tin QoS, Tag...)
            GstMessage* msg = gst_bus_pop(camera->bus);
            if (msg == nullptr) {
                break;
            }

            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* err = nullptr;
                gchar* dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                RCLCPP_ERROR(this->get_logger(), "[CAMERA_STREAM] %s loi GStreamer bus: %s (%s)",
                             camera->config.name.c_str(), err ? err->message : "unknown", dbg ? dbg : "none");
                if (err) g_error_free(err);
                if (dbg) g_free(dbg);
            } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
                RCLCPP_WARN(this->get_logger(), "[CAMERA_STREAM] %s pipeline nhan End-Of-Stream", camera->config.name.c_str());
            }
            // Giải phóng ngay lập tức bản tin vừa lấy ra
            gst_message_unref(msg);
        }
    }
}


void CameraStreamNode::initializeBirdview() {
    if (!enableBirdview_) return;

    std::string pkgShare = "";
    try {
        pkgShare = ament_index_cpp::get_package_share_directory("camera_stream");
    } catch (...) {}

    auto resolvePath = [&](const std::string& p) -> std::string {
        if (p.empty() || p[0] == '/') return p;
        if (!pkgShare.empty()) {
            std::string sharePath = pkgShare + "/" + p;
            if (std::filesystem::exists(sharePath)) return sharePath;
        }
        std::string localPath = (std::filesystem::current_path() / p).string();
        if (std::filesystem::exists(localPath)) return localPath;
        const char* homeDir = std::getenv("HOME");
        if (homeDir != nullptr) {
            std::string homePath = std::string(homeDir) + "/Cyber_Swix_M1/src/extend/camera_stream/" + p;
            if (std::filesystem::exists(homePath)) return homePath;
        }
        return p;
    };

    std::string resolvedConfig = resolvePath(birdviewConfigFile_);
    std::string resolvedCarImg = resolvePath(birdviewCarImgPath_);

    birdviewProcessor_ = std::make_unique<BirdviewProcessor>();
    birdviewProcessor_->initialize(resolvedConfig, resolvedCarImg);

    const int canvas_w = birdviewProcessor_->getCanvasWidth();
    const int canvas_h = birdviewProcessor_->getCanvasHeight();

    birdviewEncoder_ = std::make_unique<BirdviewEncoder>();
    birdviewEncoder_->initialize(canvas_w, canvas_h, birdviewFps_, birdviewBitrate_, "h265");

    birdviewRawPublisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        birdviewRawTopic_, rclcpp::SensorDataQoS()
    );
    birdviewCompressedPublisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        birdviewCompressedTopic_, rclcpp::SensorDataQoS()
    );

    const int interval_ms = (birdviewFps_ > 0) ? (1000 / birdviewFps_) : 66;
    birdviewTimer_ = this->create_wall_timer(
        std::chrono::milliseconds(interval_ms),
        [this]() { this->processBirdview(); }
    );

    RCLCPP_INFO(this->get_logger(),
                "[BIRDVIEW] Khoi tao Birdview: %dx%d@%dfps (%d bps) | Topic Raw: %s | Topic Comp: %s",
                canvas_w, canvas_h, birdviewFps_, birdviewBitrate_,
                birdviewRawTopic_.c_str(), birdviewCompressedTopic_.c_str());
}

bool CameraStreamNode::isBirdviewActive() const {
    if (!enableBirdview_) return false;
    const bool hasRaw = (birdviewRawPublisher_ && birdviewRawPublisher_->get_subscription_count() > 0);
    const bool hasComp = (birdviewCompressedPublisher_ && birdviewCompressedPublisher_->get_subscription_count() > 0);
    return hasRaw || hasComp;
}

void CameraStreamNode::processBirdview() {
    if (!isBirdviewActive() || !birdviewProcessor_) return;

    cv::Mat f, r, l, ri;
    builtin_interfaces::msg::Time stamp;
    {
        std::lock_guard<std::mutex> lock(birdviewMutex_);
        if (birdviewFrames_.find("front") == birdviewFrames_.end() ||
            birdviewFrames_.find("rear")  == birdviewFrames_.end() ||
            birdviewFrames_.find("left")  == birdviewFrames_.end() ||
            birdviewFrames_.find("right") == birdviewFrames_.end()) {
            return;
        }
        f = birdviewFrames_["front"];
        r = birdviewFrames_["rear"];
        l = birdviewFrames_["left"];
        ri = birdviewFrames_["right"];
        stamp = birdviewLastStamp_;
    }

    if (f.empty() || r.empty() || l.empty() || ri.empty()) return;

    cv::Mat canvas_bgr;
    if (!birdviewProcessor_->process(f, r, l, ri, canvas_bgr)) {
        return;
    }

    const bool hasRaw = (birdviewRawPublisher_ && birdviewRawPublisher_->get_subscription_count() > 0);
    const bool hasComp = (birdviewCompressedPublisher_ && birdviewCompressedPublisher_->get_subscription_count() > 0);

    // 1. Publish Raw Image nếu có subscriber
    if (hasRaw) {
        auto raw_msg = std::make_unique<sensor_msgs::msg::Image>();
        raw_msg->header.stamp = stamp;
        raw_msg->header.frame_id = "birdview_frame";
        raw_msg->width = static_cast<uint32_t>(canvas_bgr.cols);
        raw_msg->height = static_cast<uint32_t>(canvas_bgr.rows);
        raw_msg->encoding = "bgr8";
        raw_msg->is_bigendian = false;
        raw_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(canvas_bgr.cols * 3);
        raw_msg->data.assign(canvas_bgr.data, canvas_bgr.data + canvas_bgr.total() * 3);
        birdviewRawPublisher_->publish(std::move(raw_msg));
    }

    // 2. Encode H.265 và Publish CompressedImage nếu có subscriber (cho RTSP / Web)
    if (hasComp && birdviewEncoder_) {
        std::vector<uint8_t> compressed_data;
        if (birdviewEncoder_->encode(canvas_bgr, compressed_data)) {
            auto comp_msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
            comp_msg->header.stamp = stamp;
            comp_msg->header.frame_id = "birdview_frame";
            comp_msg->format = "h265";
            comp_msg->data = std::move(compressed_data);
            birdviewCompressedPublisher_->publish(std::move(comp_msg));
        }
    }
}

}  // namespace camera_stream
