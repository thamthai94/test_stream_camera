/**
 * *****************************************************************************
 * @file      plate_detector.hpp
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     Module phát hiện biển số xe (License Plate Detector) sử dụng
 *            TensorRT 10 (YOLOv8 / YOLOv9) trực tiếp trên buffer NVMM (NV12).
 * *****************************************************************************
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "camera_stream/mosaic_kernel.cuh"

namespace camera_stream {

class PlateDetector {
public:
    struct Config {
        std::string enginePath;
        float confThreshold = 0.45f;
        float nmsThreshold = 0.35f;
        float boxScale = 1.15f;     // Mở rộng box 15% để che phủ viền biển số
        int holdFrames = 5;
        bool enable = true;
    };

    explicit PlateDetector(const Config& config);
    ~PlateDetector();

    bool initialize();
    bool isInitialized() const { return initialized_; }
    bool isEnabled() const { return initialized_; }

    /**
     * @brief Thực hiện suy luận biển số xe từ buffer NV12 trên GPU
     * @return Danh sách các bounding box FaceBox của biển số xe
     */
    std::vector<FaceBox> detect(
        const uint8_t* d_yPlane, int yPitch,
        const uint8_t* d_uvPlane, int uvPitch,
        int width, int height,
        cudaStream_t stream
    );

    float getLastMaxScore() const { return lastMaxScore_; }
    size_t getLastDetectedCount() const { return lastDetectedCount_; }

private:
    static float calculateIoU(const FaceBox& a, const FaceBox& b);
    std::vector<FaceBox> postProcess(int width, int height, float scale, float padX, float padY);
    std::vector<FaceBox> postProcessDBNet(int width, int height, float scale, float padX, float padY);

    Config config_;
    bool initialized_ = false;
    bool isDBNet_ = false;
    float unclipRatio_ = 1.5f;

    // TensorRT 10 Components
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Tự động phát hiện kích thước input từ Engine
    std::string inputName_;
    int netWidth_ = 384;
    int netHeight_ = 384;
    size_t inputElements_ = 3 * 384 * 384;

    // Tự động phát hiện output từ Engine
    std::string outputName_;
    size_t outputElements_ = 0;
    std::vector<int64_t> outputDims_;

    // Bộ nhớ GPU
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;

    // Bộ nhớ CPU
    std::vector<float> h_output_;

    float lastMaxScore_ = 0.0f;
    size_t lastDetectedCount_ = 0;
};

}  // namespace camera_stream
