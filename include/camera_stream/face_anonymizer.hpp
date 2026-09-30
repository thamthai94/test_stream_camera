/**
 * *****************************************************************************
 * @file      face_anonymizer.hpp
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     Quản lý vòng đời inference TensorRT 10 và thuật toán Anti-Leak Tracker
 *            để che mờ khuôn mặt tự động trước khi mã hóa video.
 * *****************************************************************************
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "camera_stream/mosaic_kernel.cuh"
#include "camera_stream/plate_detector.hpp"

namespace camera_stream {

class FaceAnonymizer {
public:
    struct Config {
        // Cấu hình chung
        bool enable = true;         // Công tắc tổng (Master Enable)
        int holdFrames = 5;         // Giữ vết che mờ 5 frames chống drop
        int mosaicSize = 20;        // Kích thước ô làm mờ

        // Cấu hình phát hiện khuôn mặt (Face)
        bool enableFace = true;     // Bật/tắt riêng tính năng che mặt
        std::string enginePath;     // Đường dẫn model mặt
        float confThreshold = 0.50f;
        float nmsThreshold = 0.30f;
        float boxScale = 1.35f;     // Mở rộng box để che trán/cằm

        // Cấu hình phát hiện biển số xe (Plate)
        bool enablePlate = false;   // Bật/tắt riêng tính năng che biển số xe
        std::string plateEnginePath;
        float plateConfThreshold = 0.20f;
        float plateNmsThreshold = 0.35f;
        float plateBoxScale = 1.15f;
    };

    struct TrackedFace {
        FaceBox box;
        int ttl;                    // Time-To-Live (số frames còn hiệu lực)
    };

    explicit FaceAnonymizer(const Config& config);
    ~FaceAnonymizer();

    bool initialize();
    bool processFrame(
        uint8_t* h_yPlane, uint8_t* h_uvPlane,
        int width, int height,
        int yPitch, int uvPitch,
        int cameraId = 0,
        const std::string& cameraName = ""
    );

    // Điều khiển động lúc runtime (Thread-Safe)
    void setEnableFace(bool enable) { enableFace_.store(enable); }
    void setEnablePlate(bool enable) { enablePlate_.store(enable); }
    void setEnableAll(bool enable) {
        enableFace_.store(enable);
        enablePlate_.store(enable);
    }

    bool isFaceEnabled() const { return enableFace_.load() && isFaceReady(); }
    bool isPlateEnabled() const { return enablePlate_.load() && isPlateReady(); }

    bool isEnabled() const {
        return (enableFace_.load() && isFaceReady()) ||
               (enablePlate_.load() && isPlateReady());
    }

    bool isFaceReady() const { return faceInitialized_ && context_ != nullptr; }
    bool isPlateReady() const { return plateDetector_ != nullptr && plateDetector_->isInitialized(); }

private:
    void generatePriors();
    std::vector<FaceBox> postProcess(int cameraId, const std::string& cameraName, int frameWidth, int frameHeight);
    void updateTracker(int cameraId, const std::vector<FaceBox>& detectedBoxes);
    static float calculateIoU(const FaceBox& a, const FaceBox& b);

    Config config_;
    bool faceInitialized_ = false;

    // Cờ điều khiển động độc lập lúc runtime (Thread-Safe)
    std::atomic<bool> enableFace_{true};
    std::atomic<bool> enablePlate_{true};

    // Module phát hiện biển số xe độc lập
    std::unique_ptr<PlateDetector> plateDetector_;

    // TensorRT 10 Runtime components
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;

    // Kích thước mạng UltraFace RFB-320
    static constexpr int NET_WIDTH = 320;
    static constexpr int NET_HEIGHT = 240;
    static constexpr int NUM_PRIORS = 4420;

    // Lưu thông số chẩn đoán theo từng camera
    std::unordered_map<int, float> cameraMaxScore_;
    std::unordered_map<int, size_t> cameraDetectedCount_;

    // Bộ nhớ GPU cho NV12 Frame an toàn
    uint8_t* d_yPlane_ = nullptr;
    uint8_t* d_uvPlane_ = nullptr;
    size_t yBufferSize_ = 0;
    size_t uvBufferSize_ = 0;

    // Bộ nhớ GPU cho Inference
    float* d_input_ = nullptr;       // [3, 240, 320]
    float* d_scores_ = nullptr;      // [4420, 2]
    float* d_boxes_ = nullptr;       // [4420, 4]
    FaceBox* d_faceBoxes_ = nullptr; // Mảng GPU chứa danh sách face cần mosaic

    // Bộ nhớ CPU để đọc kết quả
    std::vector<float> h_scores_;
    std::vector<float> h_boxes_;

    // Ma trận Prior Boxes (Anchors)
    struct PriorBox {
        float cx, cy, sx, sy;
    };
    std::vector<PriorBox> priors_;

    // Bộ theo dõi bảo mật chống lộ mặt riêng cho từng Camera (Per-Camera Anti-Leak Tracker)
    std::unordered_map<int, std::vector<TrackedFace>> cameraTrackedFaces_;
    std::mutex trackerMutex_;

    // Mutex đồng bộ đa luồng giữa các camera (Thread-Safe Inference)
    std::mutex inferenceMutex_;

    static constexpr int MAX_TRACKED_FACES = 32;
};

}  // namespace camera_stream
