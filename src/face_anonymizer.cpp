/**
 * *****************************************************************************
 * @file      face_anonymizer.cpp
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     Hiện thực hóa FaceAnonymizer với TensorRT 10 và Anti-Leak Tracker.
 * *****************************************************************************
 */
#include "camera_stream/face_anonymizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace camera_stream {

namespace {

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TENSORRT] " << msg << std::endl;
        }
    }
} gLogger;

}  // namespace

FaceAnonymizer::FaceAnonymizer(const Config& config)
    : config_(config),
      enableFace_(config.enableFace),
      enablePlate_(config.enablePlate) {
    h_scores_.resize(NUM_PRIORS * 2);
    h_boxes_.resize(NUM_PRIORS * 4);
    generatePriors();
}

FaceAnonymizer::~FaceAnonymizer() {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    if (d_yPlane_ != nullptr) cudaFree(d_yPlane_);
    if (d_uvPlane_ != nullptr) cudaFree(d_uvPlane_);
    if (d_input_ != nullptr) cudaFree(d_input_);
    if (d_scores_ != nullptr) cudaFree(d_scores_);
    if (d_boxes_ != nullptr) cudaFree(d_boxes_);
    if (d_faceBoxes_ != nullptr) cudaFree(d_faceBoxes_);
}

void FaceAnonymizer::generatePriors() {
    priors_.clear();
    priors_.reserve(NUM_PRIORS);

    struct StageConfig {
        int featW, featH;
        float strideW, strideH;
        std::vector<float> minBoxes;
    };

    std::vector<StageConfig> stages = {
        {40, 30, 8.0f, 8.0f, {10.0f, 16.0f, 24.0f}},
        {20, 15, 16.0f, 16.0f, {32.0f, 48.0f}},
        {10, 8, 32.0f, 32.0f, {64.0f, 96.0f}},
        {5, 4, 64.0f, 64.0f, {128.0f, 192.0f, 256.0f}}
    };

    for (const auto& stage : stages) {
        for (int y = 0; y < stage.featH; ++y) {
            for (int x = 0; x < stage.featW; ++x) {
                for (float minBox : stage.minBoxes) {
                    float cx = (x + 0.5f) * stage.strideW / static_cast<float>(NET_WIDTH);
                    float cy = (y + 0.5f) * stage.strideH / static_cast<float>(NET_HEIGHT);
                    float sx = minBox / static_cast<float>(NET_WIDTH);
                    float sy = minBox / static_cast<float>(NET_HEIGHT);
                    priors_.push_back({cx, cy, sx, sy});
                }
            }
        }
    }
}

bool FaceAnonymizer::initialize() {
    // Tạo CUDA Stream dùng chung cho Streamer / Preprocess / Mosaic
    cudaError_t err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[FACE_ANONYMIZER] Lỗi tạo CUDA Stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaMalloc(&d_faceBoxes_, MAX_TRACKED_FACES * sizeof(FaceBox));

    // 1. Khởi tạo Face Engine nếu có file model
    if (!config_.enginePath.empty()) {
        std::ifstream file(config_.enginePath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[FACE_ANONYMIZER] Cảnh báo: Không thể mở file Face engine: " << config_.enginePath << " -> Bỏ qua Face." << std::endl;
            faceInitialized_ = false;
        } else {
            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> engineData(size);
            file.read(engineData.data(), size);
            file.close();

            runtime_.reset(nvinfer1::createInferRuntime(gLogger));
            if (runtime_) {
                engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
                if (engine_) {
                    context_.reset(engine_->createExecutionContext());
                }
            }

            if (!context_) {
                std::cerr << "[FACE_ANONYMIZER] Cảnh báo: Tạo Execution Context cho Face thất bại -> Tắt che mặt." << std::endl;
                faceInitialized_ = false;
            } else {
                size_t inputSize = 3 * NET_HEIGHT * NET_WIDTH * sizeof(float);
                size_t scoresSize = NUM_PRIORS * 2 * sizeof(float);
                size_t boxesSize = NUM_PRIORS * 4 * sizeof(float);

                cudaMalloc(&d_input_, inputSize);
                cudaMalloc(&d_scores_, scoresSize);
                cudaMalloc(&d_boxes_, boxesSize);

                int32_t nbIOTensors = engine_->getNbIOTensors();
                std::string inputTensorName = "";
                std::string scoresTensorName = "";
                std::string boxesTensorName = "";

                for (int32_t i = 0; i < nbIOTensors; ++i) {
                    const char* tensorName = engine_->getIOTensorName(i);
                    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensorName);
                    nvinfer1::Dims dims = engine_->getTensorShape(tensorName);

                    if (mode == nvinfer1::TensorIOMode::kINPUT) {
                        inputTensorName = tensorName;
                        std::cout << "[FACE_ANONYMIZER] Input: " << tensorName << std::endl;
                    } else {
                        int lastDim = (dims.nbDims > 0) ? dims.d[dims.nbDims - 1] : 0;
                        if (lastDim == 2) {
                            scoresTensorName = tensorName;
                            std::cout << "[FACE_ANONYMIZER] Scores Output: " << tensorName << std::endl;
                        } else {
                            boxesTensorName = tensorName;
                            std::cout << "[FACE_ANONYMIZER] Boxes Output: " << tensorName << std::endl;
                        }
                    }
                }

                if (!inputTensorName.empty()) {
                    context_->setInputShape(inputTensorName.c_str(), nvinfer1::Dims4{1, 3, NET_HEIGHT, NET_WIDTH});
                    context_->setTensorAddress(inputTensorName.c_str(), d_input_);
                }
                if (!scoresTensorName.empty()) {
                    context_->setTensorAddress(scoresTensorName.c_str(), d_scores_);
                }
                if (!boxesTensorName.empty()) {
                    context_->setTensorAddress(boxesTensorName.c_str(), d_boxes_);
                }

                faceInitialized_ = true;
                std::cout << "[FACE_ANONYMIZER] Khởi tạo Face Engine (TensorRT 10) thành công: " << config_.enginePath << std::endl;
            }
        }
    } else {
        faceInitialized_ = false;
        std::cout << "[FACE_ANONYMIZER] Tính năng che mặt đang TẮT qua cấu hình (Face Disabled)." << std::endl;
    }

    // 2. Khởi tạo Plate Detector nếu có file model
    if (!config_.plateEnginePath.empty()) {
        PlateDetector::Config pCfg;
        pCfg.enginePath = config_.plateEnginePath;
        pCfg.confThreshold = config_.plateConfThreshold;
        pCfg.nmsThreshold = config_.plateNmsThreshold;
        pCfg.boxScale = config_.plateBoxScale;
        pCfg.holdFrames = config_.holdFrames;

        plateDetector_ = std::make_unique<PlateDetector>(pCfg);
        if (!plateDetector_->initialize()) {
            std::cerr << "[FACE_ANONYMIZER] Cảnh báo: Khởi tạo Plate Detector thất bại -> Bỏ qua che biển số." << std::endl;
            plateDetector_.reset();
        } else {
            std::cout << "[FACE_ANONYMIZER] Khởi tạo Plate Detector thành công: " << config_.plateEnginePath << std::endl;
        }
    } else {
        std::cout << "[FACE_ANONYMIZER] Không cấu hình đường dẫn Plate Engine -> Bỏ qua che biển số xe." << std::endl;
    }

    std::cout << "[FACE_ANONYMIZER] Trạng thái mô hình: [Face: "
              << (isFaceReady() ? "SẴN SÀNG" : "CHƯA SẴN SÀNG")
              << " | Plate: "
              << (isPlateReady() ? "SẴN SÀNG" : "CHƯA SẴN SÀNG")
              << "]" << std::endl;
    std::cout << "[FACE_ANONYMIZER] Trạng thái kích hoạt hiện tại: [Face: "
              << (isFaceEnabled() ? "BẬT" : "TẮT")
              << " | Plate: "
              << (isPlateEnabled() ? "BẬT" : "TẮT")
              << "]" << std::endl;

    return true;
}

float FaceAnonymizer::calculateIoU(const FaceBox& a, const FaceBox& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float interW = std::max(0.0f, x2 - x1);
    float interH = std::max(0.0f, y2 - y1);
    float interArea = interW * interH;

    float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float unionArea = areaA + areaB - interArea;

    return (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
}

void FaceAnonymizer::updateTracker(int cameraId, const std::vector<FaceBox>& detectedBoxes) {
    std::lock_guard<std::mutex> lock(trackerMutex_);

    auto& trackedFaces = cameraTrackedFaces_[cameraId];
    std::vector<bool> matched(detectedBoxes.size(), false);

    // 1. Cập nhật vết với các khuôn mặt đã bắt được của camera này
    for (auto& tracked : trackedFaces) {
        float bestIoU = 0.0f;
        int bestIdx = -1;

        for (size_t i = 0; i < detectedBoxes.size(); ++i) {
            if (matched[i]) continue;
            float iou = calculateIoU(tracked.box, detectedBoxes[i]);
            if (iou > bestIoU && iou > 0.25f) {
                bestIoU = iou;
                bestIdx = static_cast<int>(i);
            }
        }

        if (bestIdx >= 0) {
            // Khớp với box mới -> Cập nhật tọa độ và hồi phục TTL
            tracked.box = detectedBoxes[bestIdx];
            tracked.ttl = config_.holdFrames;
            matched[bestIdx] = true;
        } else {
            // Không khớp -> Giảm TTL nhưng vẫn giữ lại che mờ (Anti-Leak)
            tracked.ttl--;
        }
    }

    // 2. Thêm các khuôn mặt mới phát hiện chưa có trong tracker của camera này
    for (size_t i = 0; i < detectedBoxes.size(); ++i) {
        if (!matched[i] && trackedFaces.size() < MAX_TRACKED_FACES) {
            trackedFaces.push_back({detectedBoxes[i], config_.holdFrames});
        }
    }

    // 3. Xóa các vết đã hết hạn TTL
    trackedFaces.erase(
        std::remove_if(trackedFaces.begin(), trackedFaces.end(),
                       [](const TrackedFace& t) { return t.ttl <= 0; }),
        trackedFaces.end()
    );
}

std::vector<FaceBox> FaceAnonymizer::postProcess(int cameraId, const std::string& cameraName, int frameWidth, int frameHeight) {
    (void)cameraName;
    std::vector<FaceBox> candidates;
    candidates.reserve(32);

    float maxScore = 0.0f;
    constexpr float VARIANCE[2] = {0.1f, 0.2f};

    for (int i = 0; i < NUM_PRIORS; ++i) {
        float s0 = h_scores_[i * 2 + 0];
        float s1 = h_scores_[i * 2 + 1];

        // Tự động kiểm tra Softmax: Nếu model xuất raw logits (tổng khác 1.0)
        float score = s1;
        if (std::abs((s0 + s1) - 1.0f) > 0.05f) {
            // Lọc nhanh: Nếu logit nền s0 áp đảo s1 thì xác suất chắc chắn < 0.12, bỏ qua std::exp()
            if (s0 - s1 > 2.0f) {
                continue;
            }
            float max_val = std::max(s0, s1);
            float exp0 = std::exp(s0 - max_val);
            float exp1 = std::exp(s1 - max_val);
            score = exp1 / (exp0 + exp1);
        }

        if (score > maxScore) {
            maxScore = score;
        }

        if (score < config_.confThreshold) {
            continue;
        }

        const auto& prior = priors_[i];
        float locX = h_boxes_[i * 4 + 0];
        float locY = h_boxes_[i * 4 + 1];
        float locW = h_boxes_[i * 4 + 2];
        float locH = h_boxes_[i * 4 + 3];

        float cx = prior.cx + locX * VARIANCE[0] * prior.sx;
        float cy = prior.cy + locY * VARIANCE[0] * prior.sy;
        float w = prior.sx * std::exp(locW * VARIANCE[1]);
        float h = prior.sy * std::exp(locH * VARIANCE[1]);

        // Mở rộng box theo scale an toàn (Safety Margin)
        float scaledW = w * config_.boxScale;
        float scaledH = h * (config_.boxScale * 1.15f); // Thêm chiều dọc để che trán và cằm

        // Dịch nhẹ tâm lên trên để che trán/tóc tốt hơn
        cy -= h * 0.05f;

        float x1 = std::max(0.0f, (cx - scaledW * 0.5f) * frameWidth);
        float y1 = std::max(0.0f, (cy - scaledH * 0.5f) * frameHeight);
        float x2 = std::min(static_cast<float>(frameWidth - 1), (cx + scaledW * 0.5f) * frameWidth);
        float y2 = std::min(static_cast<float>(frameHeight - 1), (cy + scaledH * 0.5f) * frameHeight);

        if (x2 > x1 && y2 > y1) {
            candidates.push_back({x1, y1, x2, y2, score});
        }
    }

    // Sắp xếp theo độ tin cậy giảm dần
    std::sort(candidates.begin(), candidates.end(),
              [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });

    // Non-Maximum Suppression (NMS)
    std::vector<FaceBox> nmsBoxes;
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        nmsBoxes.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && calculateIoU(candidates[i], candidates[j]) > config_.nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    cameraMaxScore_[cameraId] = maxScore;
    cameraDetectedCount_[cameraId] = candidates.size();

    return nmsBoxes;
}

bool FaceAnonymizer::processFrame(
    uint8_t* h_yPlane, uint8_t* h_uvPlane,
    int width, int height,
    int yPitch, int uvPitch,
    int cameraId,
    const std::string& cameraName
) {
    if (!isEnabled() || h_yPlane == nullptr || h_uvPlane == nullptr) {
        return false;
    }

    // Khóa Mutex tuần tự hóa để bảo đảm an toàn đa luồng giữa 4 camera
    std::lock_guard<std::mutex> lock(inferenceMutex_);

    size_t ySize = static_cast<size_t>(height) * yPitch;
    size_t uvSize = static_cast<size_t>(height / 2) * uvPitch;

    // Cấp phát/mở rộng buffer GPU cho frame nếu cần
    if (d_yPlane_ == nullptr || yBufferSize_ < ySize) {
        if (d_yPlane_) cudaFree(d_yPlane_);
        if (d_uvPlane_) cudaFree(d_uvPlane_);
        cudaMalloc(&d_yPlane_, ySize);
        cudaMalloc(&d_uvPlane_, uvSize);
        yBufferSize_ = ySize;
        uvBufferSize_ = uvSize;
    }

    // Sao chép an toàn từ mapped host/device memory sang buffer device (chỉ 460KB, mất ~0.005 ms)
    cudaMemcpyAsync(d_yPlane_, h_yPlane, ySize, cudaMemcpyDefault, stream_);
    cudaMemcpyAsync(d_uvPlane_, h_uvPlane, uvSize, cudaMemcpyDefault, stream_);

    // 100% khung hình đều chạy trực tiếp suy luận AI (Không bỏ qua frame nào)
    std::vector<FaceBox> detectedBoxes;
    detectedBoxes.reserve(16);

    // 1. Chạy phát hiện khuôn mặt nếu được bật (Face Engine)
    if (isFaceEnabled()) {
        cudaPreprocessNV12ToRGBPlanar(
            d_yPlane_, yPitch, d_uvPlane_, uvPitch,
            width, height,
            d_input_, NET_WIDTH, NET_HEIGHT,
            stream_
        );

        context_->enqueueV3(stream_);

        cudaMemcpyAsync(h_scores_.data(), d_scores_, h_scores_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_boxes_.data(), d_boxes_, h_boxes_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        std::vector<FaceBox> faceBoxes = postProcess(cameraId, cameraName, width, height);
        detectedBoxes.insert(detectedBoxes.end(), faceBoxes.begin(), faceBoxes.end());
    } else {
        cameraMaxScore_[cameraId] = 0.0f;
        cameraDetectedCount_[cameraId] = 0;
    }

    // 2. Chạy phát hiện biển số xe / chữ số nếu được bật (Plate Detector)
    if (isPlateEnabled()) {
        std::vector<FaceBox> plateBoxes = plateDetector_->detect(
            d_yPlane_, yPitch, d_uvPlane_, uvPitch, width, height, stream_
        );
        detectedBoxes.insert(detectedBoxes.end(), plateBoxes.begin(), plateBoxes.end());
    }

    // Cập nhật Anti-Leak Tracker cho riêng camera này
    updateTracker(cameraId, detectedBoxes);

    // Lấy danh sách các box cần che CỦA RIÊNG CAMERA NÀY (từ Tracker)
    std::vector<FaceBox> activeBoxes;
    {
        std::lock_guard<std::mutex> lockTracker(trackerMutex_);
        auto& trackedFaces = cameraTrackedFaces_[cameraId];
        activeBoxes.reserve(trackedFaces.size());
        for (const auto& t : trackedFaces) {
            activeBoxes.push_back(t.box);
        }
    }

    // Áp dụng CUDA Mosaic Kernel che mờ đồng thời cả Mặt và Biển số
    if (!activeBoxes.empty()) {
        cudaMemcpyAsync(d_faceBoxes_, activeBoxes.data(), activeBoxes.size() * sizeof(FaceBox), cudaMemcpyHostToDevice, stream_);
        cudaApplyMosaicNV12(
            d_yPlane_, yPitch,
            d_uvPlane_, uvPitch,
            width, height,
            d_faceBoxes_, static_cast<int>(activeBoxes.size()),
            config_.mosaicSize,
            stream_
        );
        // Sao chép kết quả đã che mờ ngược lại buffer của GStreamer
        cudaMemcpyAsync(h_yPlane, d_yPlane_, ySize, cudaMemcpyDefault, stream_);
        cudaMemcpyAsync(h_uvPlane, d_uvPlane_, uvSize, cudaMemcpyDefault, stream_);
        cudaStreamSynchronize(stream_);
        return true;
    }

    return false;
}

}  // namespace camera_stream
