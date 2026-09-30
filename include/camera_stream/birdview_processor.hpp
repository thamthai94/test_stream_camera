/**
 * *****************************************************************************
 * @file      birdview_processor.hpp
 * @author    SwiX Team
 * @date      September 2026
 * @brief     Module ghép ảnh Birdview toàn cảnh 4 camera bằng OpenCV CUDA.
 *            Tận dụng Ray-Casting 3D LUT + GPU Remap + Alpha Blending.
 * *****************************************************************************
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/calib3d.hpp>

namespace camera_stream {

struct CamParams {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.45f;
    float yaw = 0.0f;
    float pitch = 8.0f;
    float roll = 0.0f;
};

struct GlobalParams {
    float ppm = 75.0f;
    int canvas_w = 600;
    int canvas_h = 600;
    float max_angle_deg = 60.0f;
    float rmax = 4.0f;
    float fade = 1.0f;
    float dir_gamma = 1.0f;
    float robot_length = 1.6f;
    float robot_width = 0.8f;
    float car_scale = 1.35f;
};

class BirdviewProcessor {
public:
    BirdviewProcessor();
    ~BirdviewProcessor() = default;

    /**
     * @brief Khởi tạo các tham số, đọc file JSON hiệu chuẩn và ảnh overlay xe
     */
    bool initialize(const std::string& config_file, const std::string& car_img_path);

    /**
     * @brief Xử lý ghép 4 ảnh thô BGRx (CV_8UC4) thành ảnh Birdview toàn cảnh BGR (CV_8UC3)
     */
    bool process(const cv::Mat& front_bgrx,
                 const cv::Mat& rear_bgrx,
                 const cv::Mat& left_bgrx,
                 const cv::Mat& right_bgrx,
                 cv::Mat& out_canvas_bgr);

    int getCanvasWidth() const { return g_.canvas_w; }
    int getCanvasHeight() const { return g_.canvas_h; }

private:
    void loadParams(const std::string& param_file);
    void loadCarImage(const std::string& car_img_path);
    void buildAllLuts(int in_w, int in_h);
    void buildLut(const CamParams& cam, const cv::Mat& K, const cv::Mat& D,
                  int img_w, int img_h,
                  cv::cuda::GpuMat& d_mapx, cv::cuda::GpuMat& d_mapy);
    void initCleanMasks();
    void drawPngOverlay(cv::Mat& background_bgr);

    GlobalParams g_;
    CamParams cam_f_, cam_r_, cam_l_, cam_ri_;

    bool is_lut_built_ = false;

    // CUDA Remap LUTs
    cv::cuda::GpuMat d_mapx_f_, d_mapy_f_;
    cv::cuda::GpuMat d_mapx_r_, d_mapy_r_;
    cv::cuda::GpuMat d_mapx_l_, d_mapy_l_;
    cv::cuda::GpuMat d_mapx_ri_, d_mapy_ri_;

    // CUDA Blend Masks
    cv::cuda::GpuMat d_blend_mask_f_, d_blend_mask_r_, d_blend_mask_l_, d_blend_mask_ri_;

    // Tái sử dụng GPU buffers
    cv::cuda::GpuMat d_f_, d_r_, d_l_, d_ri_;
    cv::cuda::GpuMat d_warp_f_, d_warp_r_, d_warp_l_, d_warp_ri_;
    cv::cuda::GpuMat d_f_f32_, d_r_f32_, d_l_f32_, d_ri_f32_;
    cv::cuda::GpuMat d_blend_f_, d_blend_r_, d_blend_l_, d_blend_ri_;
    cv::cuda::GpuMat d_canvas_f32_, d_canvas_;

    // Tái sử dụng CPU buffers
    cv::Mat canvas_cpu_4c_;

    // Dữ liệu xe SwiX overlay
    bool car_loaded_ = false;
    cv::Rect car_roi_;
    cv::Mat car_color_weighted_;
    cv::Mat car_inv_alpha_;
};

} // namespace camera_stream
