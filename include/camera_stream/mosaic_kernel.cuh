/**
 * *****************************************************************************
 * @file      mosaic_kernel.cuh
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     Định nghĩa interface cho CUDA Mosaic Kernel và Preprocessing.
 *            Thực hiện trực tiếp trên VRAM (NV12) mà không tốn CPU.
 * *****************************************************************************
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace camera_stream {

struct FaceBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

/**
 * @brief Tiền xử lý NV12 sang float Planar RGB chuẩn hóa cho UltraFace (320x240)
 *        Chuẩn hóa: (pixel - 127.0f) / 128.0f
 */
cudaError_t cudaPreprocessNV12ToRGBPlanar(
    const uint8_t* d_yPlane, int yPitch,
    const uint8_t* d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* d_rgbPlanar, int dstWidth, int dstHeight,
    cudaStream_t stream = nullptr
);

/**
 * @brief Tiền xử lý NV12 sang float Planar RGB chuẩn hóa [0.0f, 1.0f] cho mạng YOLO (Plate Detector)
 */
cudaError_t cudaPreprocessNV12ToRGBPlanarYOLO(
    const uint8_t* d_yPlane, int yPitch,
    const uint8_t* d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* d_rgbPlanar, int dstWidth, int dstHeight,
    cudaStream_t stream = nullptr
);

/**
 * @brief Tiền xử lý Letterbox bảo toàn 100% tỉ lệ hình học, đệm màu xám 114 cho YOLO
 */
cudaError_t cudaPreprocessNV12ToRGBPlanarYOLOLetterbox(
    const uint8_t* d_yPlane, int yPitch,
    const uint8_t* d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* d_rgbPlanar, int dstWidth, int dstHeight,
    float scale, float padX, float padY,
    bool useImageNetNorm = false,
    cudaStream_t stream = nullptr
);

/**
 * @brief Làm mờ Mosaic (Pixelation) trực tiếp trên NV12 buffer (Plane Y và Plane UV)
 */
cudaError_t cudaApplyMosaicNV12(
    uint8_t* d_yPlane, int yPitch,
    uint8_t* d_uvPlane, int uvPitch,
    int width, int height,
    const FaceBox* d_boxes, int numBoxes,
    int mosaicSize,
    cudaStream_t stream = nullptr
);

}  // namespace camera_stream
