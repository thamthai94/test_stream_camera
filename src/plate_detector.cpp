/**
 * *****************************************************************************
 * @file      plate_detector.cpp
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     Hiện thực module phát hiện biển số xe (License Plate Detector)
 *            bằng TensorRT 10 với zero-copy NVMM trên GPU Jetson Orin.
 * *****************************************************************************
 */

#include "camera_stream/plate_detector.hpp"

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace camera_stream {

namespace {
class PlateTrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "[PLATE_DETECTOR_TRT] " << msg << std::endl;
        }
    }
} gPlateLogger;
}  // namespace

PlateDetector::PlateDetector(const Config& config)
    : config_(config) {}

PlateDetector::~PlateDetector() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
}

bool PlateDetector::initialize() {
    if (config_.enginePath.empty()) {
        std::cerr << "[PLATE_DETECTOR] Duong dan engine rong!" << std::endl;
        return false;
    }

    std::string finalPath = config_.enginePath;
    std::ifstream file(finalPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::vector<std::string> fallbacks = {
            "models/plate_model/yolov11n_plate_fp16.engine",
            "models/plate_model/yolov9t314_license_plate_detector_fp16.engine"
        };
        for (const auto& alt : fallbacks) {
            std::ifstream altFile(alt, std::ios::binary | std::ios::ate);
            if (altFile.is_open()) {
                finalPath = alt;
                file = std::move(altFile);
                std::cout << "[PLATE_DETECTOR] Su dung fallback engine: " << finalPath << std::endl;
                break;
            }
        }
    }

    if (!file.is_open()) {
        std::cerr << "[PLATE_DETECTOR] Khong mo duoc file engine: " << config_.enginePath << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "[PLATE_DETECTOR] Doc file engine that bai: " << config_.enginePath << std::endl;
        return false;
    }

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gPlateLogger));
    if (!runtime_) {
        std::cerr << "[PLATE_DETECTOR] Tao TensorRT Runtime that bai!" << std::endl;
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(buffer.data(), size)
    );
    if (!engine_) {
        std::cerr << "[PLATE_DETECTOR] Giai ma Engine that bai!" << std::endl;
        return false;
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "[PLATE_DETECTOR] Tao Execution Context that bai!" << std::endl;
        return false;
    }

    // Tự động phân tích các cổng Input / Output của TensorRT 10
    int nbIOTensors = engine_->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        nvinfer1::Dims dims = engine_->getTensorShape(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
            // Nếu có dynamic batch dimension
            if (dims.d[0] < 0) {
                dims.d[0] = 1;
            }
            if (dims.nbDims >= 4) {
                netHeight_ = (dims.d[2] > 0) ? dims.d[2] : 640;
                netWidth_ = (dims.d[3] > 0) ? dims.d[3] : 640;
            }
            context_->setInputShape(inputName_.c_str(), nvinfer1::Dims4{1, 3, netHeight_, netWidth_});
            inputElements_ = 3 * netHeight_ * netWidth_;
            std::cout << "[PLATE_DETECTOR] Input: " << inputName_ 
                      << " [" << 1 << "x3x" << netHeight_ << "x" << netWidth_ << "]" << std::endl;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName_ = name;
        }
    }

    if (inputName_.empty() || outputName_.empty()) {
        std::cerr << "[PLATE_DETECTOR] Khong tim thay du input/output tensor trong Engine!" << std::endl;
        return false;
    }

    // Sau khi setInputShape, lay runtime shape cua output tu context_
    nvinfer1::Dims outDims = context_->getTensorShape(outputName_.c_str());
    outputDims_.clear();
    outputElements_ = 1;
    for (int d = 0; d < outDims.nbDims; ++d) {
        int64_t dimVal = outDims.d[d];
        if (dimVal <= 0) {
            if (d == 0 && outDims.nbDims == 2) {
                dimVal = 300; // Format [-1, 7]: cap phat 300 detections
            } else if (d == 0) {
                dimVal = 1;   // Batch size = 1
            } else if (outDims.nbDims == 3 && (outDims.d[2] == 6 || outDims.d[2] == 7)) {
                dimVal = 300; // End2End max detections
            } else {
                // YOLO multi-scale grid: (H/8 * W/8) + (H/16 * W/16) + (H/32 * W/32)
                int g8  = (netHeight_ / 8)  * (netWidth_ / 8);
                int g16 = (netHeight_ / 16) * (netWidth_ / 16);
                int g32 = (netHeight_ / 32) * (netWidth_ / 32);
                dimVal = g8 + g16 + g32;
            }
        }
        outputDims_.push_back(dimVal);
        outputElements_ *= dimVal;
    }
    std::cout << "[PLATE_DETECTOR] Output: " << outputName_ << " [";
    for (size_t d = 0; d < outputDims_.size(); ++d) {
        std::cout << outputDims_[d] << (d + 1 < outputDims_.size() ? "x" : "");
    }
    std::cout << "] (Elements: " << outputElements_ << ")" << std::endl;

    // Tự động nhận diện cấu trúc mô hình: DBNet / PaddleOCR hay YOLO
    if ((outDims.nbDims == 4 && outDims.d[1] == 1) ||
        (outDims.nbDims == 3 && outDims.d[0] == 1 && outDims.d[1] >= 100 && outDims.d[2] >= 100) ||
        config_.enginePath.find("ocr") != std::string::npos) {
        isDBNet_ = true;
        outputElements_ = netHeight_ * netWidth_;
        std::cout << "[PLATE_DETECTOR] Mode: DBNet / PaddleOCR Scene Text Detector (" << netWidth_ << "x" << netHeight_ << ")" << std::endl;
    } else {
        isDBNet_ = false;
        std::cout << "[PLATE_DETECTOR] Mode: YOLO Object Detector (" << netWidth_ << "x" << netHeight_ << ")" << std::endl;
    }

    // Cấp phát bộ nhớ GPU cho input và output
    cudaMalloc(&d_input_, inputElements_ * sizeof(float));
    cudaMalloc(&d_output_, outputElements_ * sizeof(float));
    h_output_.resize(outputElements_, 0.0f);

    context_->setTensorAddress(inputName_.c_str(), d_input_);
    context_->setTensorAddress(outputName_.c_str(), d_output_);

    initialized_ = true;
    std::cout << "[PLATE_DETECTOR] Khoi tao Plate Detector thanh cong: " << config_.enginePath << std::endl;
    return true;
}

float PlateDetector::calculateIoU(const FaceBox& a, const FaceBox& b) {
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

std::vector<FaceBox> PlateDetector::detect(
    const uint8_t* d_yPlane, int yPitch,
    const uint8_t* d_uvPlane, int uvPitch,
    int width, int height,
    cudaStream_t stream
) {
    if (!isInitialized() || d_yPlane == nullptr || d_uvPlane == nullptr) {
        return {};
    }

    // 1. Tiền xử lý NV12 sang Planar RGB [0.0, 1.0] cho YOLO qua Letterbox (100% tỉ lệ thật)
    float scale = std::min(static_cast<float>(netWidth_) / width, static_cast<float>(netHeight_) / height);
    float nw = width * scale;
    float nh = height * scale;
    float padX = (netWidth_ - nw) * 0.5f;
    float padY = (netHeight_ - nh) * 0.5f;

    cudaPreprocessNV12ToRGBPlanarYOLOLetterbox(
        d_yPlane, yPitch, d_uvPlane, uvPitch,
        width, height,
        d_input_, netWidth_, netHeight_,
        scale, padX, padY,
        isDBNet_,
        stream
    );

    // 2. Chạy inference TensorRT
    context_->enqueueV3(stream);

    // 3. Đọc kết quả output về host
    cudaMemcpyAsync(h_output_.data(), d_output_, outputElements_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. Giải mã bounding boxes
    if (isDBNet_) {
        return postProcessDBNet(width, height, scale, padX, padY);
    } else {
        return postProcess(width, height, scale, padX, padY);
    }
}

std::vector<FaceBox> PlateDetector::postProcess(int width, int height, float scale, float padX, float padY) {
    std::vector<FaceBox> candidates;
    candidates.reserve(32);
    float maxScore = 0.0f;

    // Kiểm tra định dạng đầu ra của YOLO:
    // Format A: End2End NMS [N, 7] hoặc [1, N, 6] hoặc [1, N, 7]
    // Format B: Standard YOLO [1, 5, N] (cx, cy, w, h, score)
    bool isDetectionRows = false;
    int numBoxes = 0;
    int boxStride = 6;

    if (outputDims_.size() == 2 && outputDims_[1] >= 6) {
        isDetectionRows = true;
        numBoxes = static_cast<int>(outputDims_[0]);
        boxStride = static_cast<int>(outputDims_[1]);
    } else if (outputDims_.size() == 3 && (outputDims_[2] == 6 || outputDims_[2] == 7)) {
        isDetectionRows = true;
        numBoxes = static_cast<int>(outputDims_[1]);
        boxStride = static_cast<int>(outputDims_[2]);
    }

    if (isDetectionRows) {
        for (int i = 0; i < numBoxes; ++i) {
            const float* boxData = &h_output_[i * boxStride];
            if (boxData[0] == 0.0f && boxData[1] == 0.0f && boxData[2] == 0.0f && boxData[3] == 0.0f) {
                continue; // Padding
            }

            float score = 0.0f;
            float bx1 = 0.0f, by1 = 0.0f, bx2 = 0.0f, by2 = 0.0f;

            if (boxStride >= 7) {
                float sc1 = boxData[5];
                float sc2 = boxData[6];
                score = std::max(sc1, sc2);
                bx1 = boxData[1];
                by1 = boxData[2];
                bx2 = boxData[3];
                by2 = boxData[4];
                if (boxData[0] > 5.0f || (score <= 0.001f && boxData[4] > 0.001f)) {
                    bx1 = boxData[0];
                    by1 = boxData[1];
                    bx2 = boxData[2];
                    by2 = boxData[3];
                    score = boxData[4];
                }
            } else {
                bx1 = boxData[0];
                by1 = boxData[1];
                bx2 = boxData[2];
                by2 = boxData[3];
                score = boxData[4];
            }

            if (score > maxScore) maxScore = score;
            if (score < config_.confThreshold) continue;

            // Nếu tọa độ chuẩn hóa [0.0, 1.0]
            if (bx2 <= 1.05f && by2 <= 1.05f) {
                bx1 *= netWidth_;
                by1 *= netHeight_;
                bx2 *= netWidth_;
                by2 *= netHeight_;
            }

            // Hoàn tác đệm Letterbox về tọa độ ảnh gốc thật
            float origX1 = (bx1 - padX) / scale;
            float origY1 = (by1 - padY) / scale;
            float origX2 = (bx2 - padX) / scale;
            float origY2 = (by2 - padY) / scale;

            // Áp dụng scale an toàn
            float bw = origX2 - origX1;
            float bh = origY2 - origY1;
            float cx = origX1 + bw * 0.5f;
            float cy = origY1 + bh * 0.5f;
            bw *= config_.boxScale;
            bh *= config_.boxScale;

            float x1 = std::max(0.0f, cx - bw * 0.5f);
            float y1 = std::max(0.0f, cy - bh * 0.5f);
            float x2 = std::min(static_cast<float>(width - 1), cx + bw * 0.5f);
            float y2 = std::min(static_cast<float>(height - 1), cy + bh * 0.5f);

            if (x2 > x1 && y2 > y1) {
                candidates.push_back({x1, y1, x2, y2, score});
            }
        }
    } else {
        // Standard YOLO format [1, channels, numAnchors] hoặc [1, numAnchors, channels]
        int numAnchors = 0;
        int numChannels = 5;
        bool transposed = false; // transposed = true nghia la [1, numAnchors, numChannels]

        if (outputDims_.size() == 3) {
            if (outputDims_[1] < outputDims_[2]) {
                // Dang tieu chuan cua Ultralytics YOLO: [1, channels, anchors] (vi du: [1, 5, 8400])
                numChannels = static_cast<int>(outputDims_[1]);
                numAnchors = static_cast<int>(outputDims_[2]);
                transposed = false;
            } else {
                // Dang transpose: [1, anchors, channels] (vi du: [1, 8400, 5])
                numAnchors = static_cast<int>(outputDims_[1]);
                numChannels = static_cast<int>(outputDims_[2]);
                transposed = true;
            }
        } else if (outputDims_.size() == 2) {
            if (outputDims_[0] < outputDims_[1]) {
                numChannels = static_cast<int>(outputDims_[0]);
                numAnchors = static_cast<int>(outputDims_[1]);
                transposed = false;
            } else {
                numAnchors = static_cast<int>(outputDims_[0]);
                numChannels = static_cast<int>(outputDims_[1]);
                transposed = true;
            }
        }

        for (int i = 0; i < numAnchors; ++i) {
            float cx, cy, w, h;
            float score = 0.0f;
            if (!transposed) {
                cx = h_output_[0 * numAnchors + i];
                cy = h_output_[1 * numAnchors + i];
                w  = h_output_[2 * numAnchors + i];
                h  = h_output_[3 * numAnchors + i];
                for (int c = 4; c < numChannels; ++c) {
                    float s = h_output_[c * numAnchors + i];
                    if (s > score) score = s;
                }
            } else {
                cx = h_output_[i * numChannels + 0];
                cy = h_output_[i * numChannels + 1];
                w  = h_output_[i * numChannels + 2];
                h  = h_output_[i * numChannels + 3];
                for (int c = 4; c < numChannels; ++c) {
                    float s = h_output_[i * numChannels + c];
                    if (s > score) score = s;
                }
            }

            if (score > maxScore) maxScore = score;
            if (score < config_.confThreshold) continue;

            if (w <= 1.05f && h <= 1.05f) {
                cx *= netWidth_;
                cy *= netHeight_;
                w  *= netWidth_;
                h  *= netHeight_;
            }

            float bx1 = cx - w * 0.5f;
            float by1 = cy - h * 0.5f;
            float bx2 = cx + w * 0.5f;
            float by2 = cy + h * 0.5f;

            float origX1 = (bx1 - padX) / scale;
            float origY1 = (by1 - padY) / scale;
            float origX2 = (bx2 - padX) / scale;
            float origY2 = (by2 - padY) / scale;

            float bw = (origX2 - origX1) * config_.boxScale;
            float bh = (origY2 - origY1) * config_.boxScale;
            float midX = origX1 + (origX2 - origX1) * 0.5f;
            float midY = origY1 + (origY2 - origY1) * 0.5f;

            float x1 = std::max(0.0f, midX - bw * 0.5f);
            float y1 = std::max(0.0f, midY - bh * 0.5f);
            float x2 = std::min(static_cast<float>(width - 1), midX + bw * 0.5f);
            float y2 = std::min(static_cast<float>(height - 1), midY + bh * 0.5f);

            if (x2 > x1 && y2 > y1) {
                candidates.push_back({x1, y1, x2, y2, score});
            }
        }
    }

    lastMaxScore_ = maxScore;

    // NMS Suppression
    std::sort(candidates.begin(), candidates.end(),
              [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });

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

    lastDetectedCount_ = nmsBoxes.size();
    return nmsBoxes;
}

std::vector<FaceBox> PlateDetector::postProcessDBNet(int width, int height, float scale, float padX, float padY) {
    std::vector<FaceBox> results;

    cv::Mat probMap(netHeight_, netWidth_, CV_32FC1, h_output_.data());
    float thresh = (config_.confThreshold > 0.0f) ? config_.confThreshold : 0.20f;

    // Early Exit: Nếu điểm cao nhất toàn ảnh nhỏ hơn ngưỡng -> Không có biển số, thoát ngay!
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(probMap, &minVal, &maxVal);
    if (maxVal < thresh) {
        lastMaxScore_ = static_cast<float>(maxVal);
        lastDetectedCount_ = 0;
        return results;
    }

    cv::Mat binaryMap;
    cv::threshold(probMap, binaryMap, thresh, 1.0, cv::THRESH_BINARY);
    binaryMap.convertTo(binaryMap, CV_8UC1, 255.0);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryMap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 10.0) continue;

        double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.001) continue;

        double unclipDist = area * unclipRatio_ / perimeter;
        cv::RotatedRect rect = cv::minAreaRect(contour);

        rect.size.width  += 2.0f * static_cast<float>(unclipDist);
        rect.size.height += 2.0f * static_cast<float>(unclipDist);

        cv::Rect br = rect.boundingRect();

        // Hoàn tác đệm Letterbox về tọa độ ảnh gốc thật
        float origX1 = (br.x - padX) / scale;
        float origY1 = (br.y - padY) / scale;
        float origX2 = (br.x + br.width - padX) / scale;
        float origY2 = (br.y + br.height - padY) / scale;

        float bw = origX2 - origX1;
        float bh = origY2 - origY1;
        float cx = origX1 + bw * 0.5f;
        float cy = origY1 + bh * 0.5f;
        bw *= config_.boxScale;
        bh *= config_.boxScale;

        float x1 = std::max(0.0f, cx - bw * 0.5f);
        float y1 = std::max(0.0f, cy - bh * 0.5f);
        float x2 = std::min(static_cast<float>(width - 1), cx + bw * 0.5f);
        float y2 = std::min(static_cast<float>(height - 1), cy + bh * 0.5f);

        if (x2 > x1 + 4.0f && y2 > y1 + 4.0f) {
            results.push_back({x1, y1, x2, y2, 0.90f});
        }
    }

    lastMaxScore_ = results.empty() ? 0.0f : 0.90f;
    lastDetectedCount_ = results.size();
    return results;
}

}  // namespace camera_stream
