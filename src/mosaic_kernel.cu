/**
 * *****************************************************************************
 * @file      mosaic_kernel.cu
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     CUDA Kernel xử lý tiền xử lý và che mờ trực tiếp trên NV12 buffer.
 * *****************************************************************************
 */
#include "camera_stream/mosaic_kernel.cuh"
#include <device_launch_parameters.h>

namespace camera_stream {

// =============================================================================
// KERNEL 1: TIỀN XỬ LÝ NV12 -> PLANAR RGB (NCHW) CHUẨN HÓA CHO TENSORRT
// =============================================================================
__global__ void preprocessNV12ToRGBPlanarKernel(
    const uint8_t* __restrict__ d_yPlane, int yPitch,
    const uint8_t* __restrict__ d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* __restrict__ d_rgbPlanar, int dstWidth, int dstHeight
) {
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight) {
        return;
    }

    // Ánh xạ tọa độ từ dst (320x240) về src (640x480)
    int srcX = dstX * srcWidth / dstWidth;
    int srcY = dstY * srcHeight / dstHeight;

    if (srcX >= srcWidth) srcX = srcWidth - 1;
    if (srcY >= srcHeight) srcY = srcHeight - 1;

    // 1. Đọc giá trị Y từ plane Y
    float Y = static_cast<float>(d_yPlane[srcY * yPitch + srcX]);

    // 2. Đọc giá trị U, V từ plane UV (Interleaved)
    int uvRow = srcY / 2;
    int uvCol = srcX & ~1;  // Làm tròn chẵn
    float U = static_cast<float>(d_uvPlane[uvRow * uvPitch + uvCol]);
    float V = static_cast<float>(d_uvPlane[uvRow * uvPitch + uvCol + 1]);

    // 3. Chuyển đổi YUV (BT.601) sang RGB
    float u_diff = U - 128.0f;
    float v_diff = V - 128.0f;
    float R = Y + 1.402f * v_diff;
    float G = Y - 0.344136f * u_diff - 0.714136f * v_diff;
    float B = Y + 1.772f * u_diff;

    // Clamp [0, 255]
    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    // 4. Chuẩn hóa UltraFace: (x - 127.0) / 128.0
    float normR = (R - 127.0f) / 128.0f;
    float normG = (G - 127.0f) / 128.0f;
    float normB = (B - 127.0f) / 128.0f;

    // 5. Lưu vào mảng planar NCHW [3, 240, 320]
    int planeSize = dstWidth * dstHeight;
    int pixelIdx = dstY * dstWidth + dstX;
    d_rgbPlanar[0 * planeSize + pixelIdx] = normR;
    d_rgbPlanar[1 * planeSize + pixelIdx] = normG;
    d_rgbPlanar[2 * planeSize + pixelIdx] = normB;
}

// =============================================================================
// KERNEL 2: LÀM MỜ MOSAIC (PIXELATION) TRỰC TIẾP TRÊN BUFFER NV12
// =============================================================================
__global__ void mosaicNV12Kernel(
    uint8_t* d_yPlane, int yPitch,
    uint8_t* d_uvPlane, int uvPitch,
    int width, int height,
    const FaceBox* d_boxes, int numBoxes, int mosaicSize
) {
    int boxIdx = blockIdx.z;
    if (boxIdx >= numBoxes) {
        return;
    }

    const FaceBox& box = d_boxes[boxIdx];

    int x1 = max(0, static_cast<int>(box.x1));
    int y1 = max(0, static_cast<int>(box.y1));
    int x2 = min(width - 1, static_cast<int>(box.x2));
    int y2 = min(height - 1, static_cast<int>(box.y2));

    if (x1 >= x2 || y1 >= y2) {
        return;
    }

    int boxW = x2 - x1 + 1;
    int boxH = y2 - y1 + 1;

    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;

    if (localX >= boxW || localY >= boxH) {
        return;
    }

    int currentX = x1 + localX;
    int currentY = y1 + localY;

    if (currentX >= width || currentY >= height) {
        return;
    }

    // Tọa độ đại diện góc trên bên trái của ô Mosaic
    int blockX = x1 + (localX / mosaicSize) * mosaicSize;
    int blockY = y1 + (localY / mosaicSize) * mosaicSize;

    if (blockX >= width) blockX = width - 1;
    if (blockY >= height) blockY = height - 1;

    // 1. Áp dụng Mosaic cho mặt phẳng Y
    uint8_t sampleY = d_yPlane[blockY * yPitch + blockX];
    d_yPlane[currentY * yPitch + currentX] = sampleY;

    // 2. Áp dụng Mosaic cho mặt phẳng UV (Chỉ thực hiện cho các pixel chẵn để tránh race condition)
    if ((currentX % 2 == 0) && (currentY % 2 == 0)) {
        int sampleUvCol = blockX & ~1;
        int sampleUvRow = blockY / 2;

        if (sampleUvRow >= height / 2) sampleUvRow = (height / 2) - 1;
        if (sampleUvCol >= width - 1) sampleUvCol = (width - 2);

        uint8_t sampleU = d_uvPlane[sampleUvRow * uvPitch + sampleUvCol];
        uint8_t sampleV = d_uvPlane[sampleUvRow * uvPitch + sampleUvCol + 1];

        int curUvCol = currentX;
        int curUvRow = currentY / 2;

        if (curUvRow < (height / 2) && curUvCol < width) {
            d_uvPlane[curUvRow * uvPitch + curUvCol] = sampleU;
            if (curUvCol + 1 < width) {
                d_uvPlane[curUvRow * uvPitch + curUvCol + 1] = sampleV;
            }
        }
    }
}

// =============================================================================
// WRAPPERS GỌI TỪ C++
// =============================================================================
cudaError_t cudaPreprocessNV12ToRGBPlanar(
    const uint8_t* d_yPlane, int yPitch,
    const uint8_t* d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* d_rgbPlanar, int dstWidth, int dstHeight,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);

    preprocessNV12ToRGBPlanarKernel<<<grid, block, 0, stream>>>(
        d_yPlane, yPitch, d_uvPlane, uvPitch,
        srcWidth, srcHeight,
        d_rgbPlanar, dstWidth, dstHeight
    );

    return cudaGetLastError();
}

// =============================================================================
// KERNEL TIỀN XỬ LÝ NV12 -> PLANAR RGB (NCHW) CHUẨN HÓA [0.0, 1.0] CHO YOLO
// =============================================================================
__global__ void preprocessNV12ToRGBPlanarYOLOKernel(
    const uint8_t* __restrict__ d_yPlane, int yPitch,
    const uint8_t* __restrict__ d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* __restrict__ d_rgbPlanar, int dstWidth, int dstHeight
) {
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight) {
        return;
    }

    // Ánh xạ tọa độ từ dst về src
    int srcX = dstX * srcWidth / dstWidth;
    int srcY = dstY * srcHeight / dstHeight;

    if (srcX >= srcWidth) srcX = srcWidth - 1;
    if (srcY >= srcHeight) srcY = srcHeight - 1;

    // 1. Đọc giá trị Y
    float Y = static_cast<float>(d_yPlane[srcY * yPitch + srcX]);

    // 2. Đọc giá trị U, V từ UV interleaved
    int uvRow = srcY / 2;
    int uvCol = srcX & ~1;
    float U = static_cast<float>(d_uvPlane[uvRow * uvPitch + uvCol]);
    float V = static_cast<float>(d_uvPlane[uvRow * uvPitch + uvCol + 1]);

    // 3. Chuyển đổi BT.601 sang RGB
    float u_diff = U - 128.0f;
    float v_diff = V - 128.0f;
    float R = Y + 1.402f * v_diff;
    float G = Y - 0.344136f * u_diff - 0.714136f * v_diff;
    float B = Y + 1.772f * u_diff;

    // Clamp [0, 255]
    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    // 4. Chuẩn hóa YOLO: x / 255.0f
    float normR = R * (1.0f / 255.0f);
    float normG = G * (1.0f / 255.0f);
    float normB = B * (1.0f / 255.0f);

    // 5. Lưu vào mảng planar NCHW [3, dstHeight, dstWidth]
    int planeSize = dstWidth * dstHeight;
    int pixelIdx = dstY * dstWidth + dstX;
    d_rgbPlanar[0 * planeSize + pixelIdx] = normR;
    d_rgbPlanar[1 * planeSize + pixelIdx] = normG;
    d_rgbPlanar[2 * planeSize + pixelIdx] = normB;
}

cudaError_t cudaPreprocessNV12ToRGBPlanarYOLO(
    const uint8_t* d_yPlane, int yPitch,
    const uint8_t* d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* d_rgbPlanar, int dstWidth, int dstHeight,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);

    preprocessNV12ToRGBPlanarYOLOKernel<<<grid, block, 0, stream>>>(
        d_yPlane, yPitch, d_uvPlane, uvPitch,
        srcWidth, srcHeight,
        d_rgbPlanar, dstWidth, dstHeight
    );

    return cudaGetLastError();
}

// =============================================================================
// KERNEL TIỀN XỬ LÝ LETTERBOX (BẢO TOÀN TỈ LỆ 100%, ĐỆM XÁM 114) CHO YOLO
// =============================================================================
__global__ void preprocessNV12ToRGBPlanarYOLOLetterboxKernel(
    const uint8_t* __restrict__ d_yPlane, int yPitch,
    const uint8_t* __restrict__ d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* __restrict__ d_rgbPlanar, int dstWidth, int dstHeight,
    float scale, float padX, float padY,
    bool useImageNetNorm
) {
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight) {
        return;
    }

    int planeSize = dstWidth * dstHeight;
    int pixelIdx = dstY * dstWidth + dstX;

    float unpadX = dstX - padX;
    float unpadY = dstY - padY;

    // Nếu nằm trong vùng viền đệm letterbox
    if (unpadX < 0.0f || unpadX >= (srcWidth * scale) || unpadY < 0.0f || unpadY >= (srcHeight * scale)) {
        float padValR = useImageNetNorm ? -0.16568f : 0.4470588f;
        float padValG = useImageNetNorm ? -0.04018f : 0.4470588f;
        float padValB = useImageNetNorm ? 0.18248f  : 0.4470588f;
        d_rgbPlanar[0 * planeSize + pixelIdx] = padValR;
        d_rgbPlanar[1 * planeSize + pixelIdx] = padValG;
        d_rgbPlanar[2 * planeSize + pixelIdx] = padValB;
        return;
    }

    // Ánh xạ tọa độ chuẩn xác không méo hình
    int srcX = __float2int_rz(unpadX / scale);
    int srcY = __float2int_rz(unpadY / scale);

    if (srcX >= srcWidth) srcX = srcWidth - 1;
    if (srcY >= srcHeight) srcY = srcHeight - 1;

    float Y = static_cast<float>(d_yPlane[srcY * yPitch + srcX]);

    int uvRow = srcY / 2;
    int uvCol = srcX & ~1;
    float U = static_cast<float>(d_uvPlane[uvRow * uvPitch + uvCol]);
    float V = static_cast<float>(d_uvPlane[uvRow * uvPitch + uvCol + 1]);

    float u_diff = U - 128.0f;
    float v_diff = V - 128.0f;
    float R = Y + 1.402f * v_diff;
    float G = Y - 0.344136f * u_diff - 0.714136f * v_diff;
    float B = Y + 1.772f * u_diff;

    R = fminf(fmaxf(R, 0.0f), 255.0f) * (1.0f / 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f) * (1.0f / 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f) * (1.0f / 255.0f);

    if (useImageNetNorm) {
        R = (R - 0.485f) * (1.0f / 0.229f);
        G = (G - 0.456f) * (1.0f / 0.224f);
        B = (B - 0.406f) * (1.0f / 0.225f);
    }

    d_rgbPlanar[0 * planeSize + pixelIdx] = R;
    d_rgbPlanar[1 * planeSize + pixelIdx] = G;
    d_rgbPlanar[2 * planeSize + pixelIdx] = B;
}

cudaError_t cudaPreprocessNV12ToRGBPlanarYOLOLetterbox(
    const uint8_t* d_yPlane, int yPitch,
    const uint8_t* d_uvPlane, int uvPitch,
    int srcWidth, int srcHeight,
    float* d_rgbPlanar, int dstWidth, int dstHeight,
    float scale, float padX, float padY,
    bool useImageNetNorm,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);

    preprocessNV12ToRGBPlanarYOLOLetterboxKernel<<<grid, block, 0, stream>>>(
        d_yPlane, yPitch, d_uvPlane, uvPitch,
        srcWidth, srcHeight,
        d_rgbPlanar, dstWidth, dstHeight,
        scale, padX, padY,
        useImageNetNorm
    );

    return cudaGetLastError();
}

cudaError_t cudaApplyMosaicNV12(
    uint8_t* d_yPlane, int yPitch,
    uint8_t* d_uvPlane, int uvPitch,
    int width, int height,
    const FaceBox* d_boxes, int numBoxes,
    int mosaicSize,
    cudaStream_t stream
) {
    if (numBoxes <= 0 || d_boxes == nullptr) {
        return cudaSuccess;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, numBoxes);

    mosaicNV12Kernel<<<grid, block, 0, stream>>>(
        d_yPlane, yPitch, d_uvPlane, uvPitch,
        width, height,
        d_boxes, numBoxes, mosaicSize
    );

    return cudaGetLastError();
}

}  // namespace camera_stream
