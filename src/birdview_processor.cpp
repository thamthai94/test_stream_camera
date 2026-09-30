/**
 * *****************************************************************************
 * @file      birdview_processor.cpp
 * @author    SwiX Team
 * @date      September 2026
 * @brief     Cài đặt thuật toán ghép ảnh Birdview bằng OpenCV CUDA.
 * *****************************************************************************
 */
#include "camera_stream/birdview_processor.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace camera_stream {

using json = nlohmann::json;

BirdviewProcessor::BirdviewProcessor() {
    g_ = { 75.0f, 600, 600, 60.0f, 4.0f, 1.0f, 1.0f, 1.6f, 0.8f, 1.35f };
    cam_f_  = {  0.5f,  0.0f, 0.45f,   0.0f, 8.0f, 0.0f };
    cam_r_  = { -0.5f,  0.0f, 0.45f, 180.0f, 8.0f, 0.0f };
    cam_l_  = {  0.0f,  0.35f, 0.45f,  90.0f, 8.0f, 0.0f };
    cam_ri_ = {  0.0f, -0.35f, 0.45f, -90.0f, 8.0f, 0.0f };
}

bool BirdviewProcessor::initialize(const std::string& config_file, const std::string& car_img_path) {
    loadParams(config_file);
    loadCarImage(car_img_path);
    return true;
}

void BirdviewProcessor::loadParams(const std::string& param_file) {
    std::ifstream f(param_file);
    if (!f.is_open()) {
        std::cerr << "[BIRDVIEW] Khong the mo file tham so: " << param_file << " -> dung thong so mac dinh.\n";
        return;
    }

    try {
        json data;
        f >> data;
        if (data.contains("global")) {
            auto gl = data["global"];
            if (gl.contains("ppm")) g_.ppm = gl["ppm"];
            if (gl.contains("canvas_w")) g_.canvas_w = gl["canvas_w"];
            if (gl.contains("canvas_h")) g_.canvas_h = gl["canvas_h"];
            if (gl.contains("robot_length")) g_.robot_length = gl["robot_length"];
            if (gl.contains("robot_width")) g_.robot_width = gl["robot_width"];
            if (gl.contains("car_scale")) g_.car_scale = gl["car_scale"];
        }

        auto parse_cam = [](const json& c, CamParams& p) {
            if (c.contains("x")) p.x = c["x"];
            if (c.contains("y")) p.y = c["y"];
            if (c.contains("z")) p.z = c["z"];
            if (c.contains("yaw")) p.yaw = c["yaw"];
            if (c.contains("pitch")) p.pitch = c["pitch"];
            if (c.contains("roll")) p.roll = c["roll"];
        };

        if (data.contains("cameras")) {
            if (data["cameras"].contains("front")) parse_cam(data["cameras"]["front"], cam_f_);
            if (data["cameras"].contains("rear"))  parse_cam(data["cameras"]["rear"], cam_r_);
            if (data["cameras"].contains("left"))  parse_cam(data["cameras"]["left"], cam_l_);
            if (data["cameras"].contains("right")) parse_cam(data["cameras"]["right"], cam_ri_);
        }
        std::cout << "[BIRDVIEW] Da nap thanh cong file cau hinh: " << param_file
                  << " (Canvas: " << g_.canvas_w << "x" << g_.canvas_h << ", PPM: " << g_.ppm << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "[BIRDVIEW] Loi phan tich file JSON: " << e.what() << "\n";
    }
}

void BirdviewProcessor::loadCarImage(const std::string& car_img_path) {
    cv::Mat raw_img = cv::imread(car_img_path, cv::IMREAD_UNCHANGED);
    if (raw_img.empty()) {
        std::cerr << "[BIRDVIEW] Canh bao: Khong tim thay anh PNG xe tai: " << car_img_path << "\n";
        car_loaded_ = false;
        return;
    }

    int draw_w = std::round(g_.robot_width * g_.ppm * g_.car_scale);
    int draw_h = std::round(g_.robot_length * g_.ppm * g_.car_scale);
    cv::Mat resized;
    cv::resize(raw_img, resized, cv::Size(draw_w, draw_h));

    int draw_x = (g_.canvas_w - draw_w) / 2;
    int draw_y = (g_.canvas_h - draw_h) / 2;
    car_roi_ = cv::Rect(draw_x, draw_y, draw_w, draw_h);

    cv::Mat alpha, color;
    if (resized.channels() == 4) {
        std::vector<cv::Mat> fg_channels;
        cv::split(resized, fg_channels);
        alpha = fg_channels[3];
        cv::merge(std::vector<cv::Mat>{fg_channels[0], fg_channels[1], fg_channels[2]}, color);
    } else if (resized.channels() == 3) {
        color = resized.clone();
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, alpha, 5, 255, cv::THRESH_BINARY);
    } else {
        car_loaded_ = false;
        return;
    }

    cv::Mat alpha_32f;
    alpha.convertTo(alpha_32f, CV_32F, 1.0 / 255.0);
    cv::Mat inv_alpha = 1.0f - alpha_32f;

    std::vector<cv::Mat> alpha_channels{alpha_32f, alpha_32f, alpha_32f};
    std::vector<cv::Mat> inv_alpha_channels{inv_alpha, inv_alpha, inv_alpha};
    cv::Mat alpha_3c;
    cv::merge(alpha_channels, alpha_3c);
    cv::merge(inv_alpha_channels, car_inv_alpha_);

    cv::Mat color_32f;
    color.convertTo(color_32f, CV_32F);
    car_color_weighted_ = color_32f.mul(alpha_3c);
    car_loaded_ = true;
    std::cout << "[BIRDVIEW] Da nap thanh cong anh overlay xe: " << car_img_path << "\n";
}

void BirdviewProcessor::buildLut(const CamParams& cam, const cv::Mat& K, const cv::Mat& D,
                                 int img_w, int img_h,
                                 cv::cuda::GpuMat& d_mapx, cv::cuda::GpuMat& d_mapy) {
    cv::Mat map_x(g_.canvas_h, g_.canvas_w, CV_32F, cv::Scalar(-1));
    cv::Mat map_y(g_.canvas_h, g_.canvas_w, CV_32F, cv::Scalar(-1));

    float yaw = cam.yaw * static_cast<float>(CV_PI) / 180.0f;
    float pitch = cam.pitch * static_cast<float>(CV_PI) / 180.0f;
    float roll = cam.roll * static_cast<float>(CV_PI) / 180.0f;

    cv::Vec3f f(std::cos(pitch)*std::cos(yaw), std::cos(pitch)*std::sin(yaw), -std::sin(pitch));
    cv::Vec3f r(std::sin(yaw), -std::cos(yaw), 0.0f);
    cv::Vec3f d = f.cross(r);
    cv::Vec3f r2 = r * std::cos(roll) + d * std::sin(roll);
    cv::Vec3f d2 = -r * std::sin(roll) + d * std::cos(roll);

    cv::Mat R(3, 3, CV_32F);
    R.at<float>(0,0) = r2[0]; R.at<float>(1,0) = r2[1]; R.at<float>(2,0) = r2[2];
    R.at<float>(0,1) = d2[0]; R.at<float>(1,1) = d2[1]; R.at<float>(2,1) = d2[2];
    R.at<float>(0,2) = f[0];  R.at<float>(1,2) = f[1];  R.at<float>(2,2) = f[2];
    cv::Mat R_t = R.t();

    std::vector<cv::Point3f> pts3d;
    std::vector<int> valid_idx;

    for (int row = 0; row < g_.canvas_h; ++row) {
        for (int col = 0; col < g_.canvas_w; ++col) {
            float X = (g_.canvas_h / 2.0f - row) / g_.ppm;
            float Y = (g_.canvas_w / 2.0f - col) / g_.ppm;

            cv::Mat P_vec = (cv::Mat_<float>(3,1) << X - cam.x, Y - cam.y, 0.0f - cam.z);
            cv::Mat Pc_mat = R_t * P_vec;
            cv::Vec3f Pc(Pc_mat.at<float>(0,0), Pc_mat.at<float>(1,0), Pc_mat.at<float>(2,0));

            if (Pc[2] <= 1e-3f) continue;
            pts3d.push_back(cv::Point3f(Pc[0], Pc[1], Pc[2]));
            valid_idx.push_back(row * g_.canvas_w + col);
        }
    }

    std::vector<cv::Point2f> pts2d;
    if (!pts3d.empty()) {
        cv::projectPoints(pts3d, cv::Vec3f(0,0,0), cv::Vec3f(0,0,0), K, D, pts2d);
    }

    for (size_t i = 0; i < pts2d.size(); ++i) {
        int row = valid_idx[i] / g_.canvas_w;
        int col = valid_idx[i] % g_.canvas_w;
        float u = pts2d[i].x;
        float v = pts2d[i].y;

        if (u >= 0 && u < img_w - 1 && v >= 0 && v < img_h - 1) {
            map_x.at<float>(row, col) = u;
            map_y.at<float>(row, col) = v;
        }
    }

    d_mapx.upload(map_x);
    d_mapy.upload(map_y);
}

void BirdviewProcessor::initCleanMasks() {
    int W = g_.canvas_w;
    int H = g_.canvas_h;
    int CAR_W = static_cast<int>(g_.robot_width * g_.ppm);
    int CAR_H = static_cast<int>(g_.robot_length * g_.ppm);
    int CAR_X = (W - CAR_W) / 2;
    int CAR_Y = (H - CAR_H) / 2;

    cv::Mat mask_f = cv::Mat::zeros(H, W, CV_32FC1);
    cv::Mat mask_r = cv::Mat::zeros(H, W, CV_32FC1);
    cv::Mat mask_l = cv::Mat::zeros(H, W, CV_32FC1);
    cv::Mat mask_ri = cv::Mat::zeros(H, W, CV_32FC1);

    std::vector<cv::Point> poly_f = { cv::Point(0, 0), cv::Point(W, 0), cv::Point(CAR_X + CAR_W, CAR_Y), cv::Point(CAR_X, CAR_Y) };
    std::vector<cv::Point> poly_r = { cv::Point(CAR_X, CAR_Y + CAR_H), cv::Point(CAR_X + CAR_W, CAR_Y + CAR_H), cv::Point(W, H), cv::Point(0, H) };
    std::vector<cv::Point> poly_l = { cv::Point(0, 0), cv::Point(CAR_X, CAR_Y), cv::Point(CAR_X, CAR_Y + CAR_H), cv::Point(0, H) };
    std::vector<cv::Point> poly_ri = { cv::Point(W, 0), cv::Point(W, H), cv::Point(CAR_X + CAR_W, CAR_Y + CAR_H), cv::Point(CAR_X + CAR_W, CAR_Y) };

    cv::fillPoly(mask_f, std::vector<std::vector<cv::Point>>{poly_f}, cv::Scalar(1.0));
    cv::fillPoly(mask_r, std::vector<std::vector<cv::Point>>{poly_r}, cv::Scalar(1.0));
    cv::fillPoly(mask_l, std::vector<std::vector<cv::Point>>{poly_l}, cv::Scalar(1.0));
    cv::fillPoly(mask_ri, std::vector<std::vector<cv::Point>>{poly_ri}, cv::Scalar(1.0));

    int blur_size = 31;
    cv::GaussianBlur(mask_f, mask_f, cv::Size(blur_size, blur_size), 0);
    cv::GaussianBlur(mask_r, mask_r, cv::Size(blur_size, blur_size), 0);
    cv::GaussianBlur(mask_l, mask_l, cv::Size(blur_size, blur_size), 0);
    cv::GaussianBlur(mask_ri, mask_ri, cv::Size(blur_size, blur_size), 0);

    cv::Mat sum_mask = mask_f + mask_r + mask_l + mask_ri;
    cv::add(sum_mask, 1e-5, sum_mask);
    cv::divide(mask_f, sum_mask, mask_f);
    cv::divide(mask_r, sum_mask, mask_r);
    cv::divide(mask_l, sum_mask, mask_l);
    cv::divide(mask_ri, sum_mask, mask_ri);

    auto upload_wt = [](const cv::Mat& w_1c, cv::cuda::GpuMat& d_w_4c) {
        cv::Mat w_4c;
        cv::merge(std::vector<cv::Mat>{w_1c, w_1c, w_1c, w_1c}, w_4c);
        d_w_4c.upload(w_4c);
    };

    upload_wt(mask_f, d_blend_mask_f_);
    upload_wt(mask_r, d_blend_mask_r_);
    upload_wt(mask_l, d_blend_mask_l_);
    upload_wt(mask_ri, d_blend_mask_ri_);
}

void BirdviewProcessor::buildAllLuts(int in_w, int in_h) {
    float sx = static_cast<float>(in_w) / 1920.0f;
    float sy = static_cast<float>(in_h) / 1536.0f;

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        1188.3623519772 * sx, 0.0,                  925.4093602553 * sx,
        0.0,                  1188.3807315122 * sy, 736.1542573607 * sy,
        0.0,                  0.0,                  1.0);

    cv::Mat D = (cv::Mat_<double>(1, 8) <<
        6.8963005101,  4.0036274878, 0.0000473174, -0.0000035493,
        0.2049163368, 7.3181577843, 6.8436552413, 1.1877068823);

    buildLut(cam_f_, K, D, in_w, in_h, d_mapx_f_, d_mapy_f_);
    buildLut(cam_r_, K, D, in_w, in_h, d_mapx_r_, d_mapy_r_);
    buildLut(cam_l_, K, D, in_w, in_h, d_mapx_l_, d_mapy_l_);
    buildLut(cam_ri_, K, D, in_w, in_h, d_mapx_ri_, d_mapy_ri_);

    initCleanMasks();
    is_lut_built_ = true;
    std::cout << "[BIRDVIEW] Da tinh toan xong Ray-Casting 3D LUT (Input: " << in_w << "x" << in_h << ")!\n";
}

void BirdviewProcessor::drawPngOverlay(cv::Mat& background_bgr) {
    if (!car_loaded_ || car_color_weighted_.empty()) return;
    cv::Mat bg_roi = background_bgr(car_roi_);
    cv::Mat bg_32f;
    bg_roi.convertTo(bg_32f, CV_32F);
    cv::Mat result = car_color_weighted_ + bg_32f.mul(car_inv_alpha_);
    result.convertTo(bg_roi, CV_8UC3);
}

bool BirdviewProcessor::process(const cv::Mat& front_bgrx,
                                const cv::Mat& rear_bgrx,
                                const cv::Mat& left_bgrx,
                                const cv::Mat& right_bgrx,
                                cv::Mat& out_canvas_bgr) {
    if (front_bgrx.empty() || rear_bgrx.empty() || left_bgrx.empty() || right_bgrx.empty()) {
        return false;
    }

    if (!is_lut_built_) {
        buildAllLuts(front_bgrx.cols, front_bgrx.rows);
    }

    // 1. Upload ảnh 4 kênh BGRx (CV_8UC4) lên GPU VRAM
    d_f_.upload(front_bgrx);
    d_r_.upload(rear_bgrx);
    d_l_.upload(left_bgrx);
    d_ri_.upload(right_bgrx);

    // 2. CUDA Remap trực tiếp trên 4 kênh BGRx
    cv::cuda::remap(d_f_, d_warp_f_, d_mapx_f_, d_mapy_f_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::cuda::remap(d_r_, d_warp_r_, d_mapx_r_, d_mapy_r_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::cuda::remap(d_l_, d_warp_l_, d_mapx_l_, d_mapy_l_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::cuda::remap(d_ri_, d_warp_ri_, d_mapx_ri_, d_mapy_ri_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // 3. Chuyển sang Float32 (CV_32FC4) để nhân ma trận trọng số
    d_warp_f_.convertTo(d_f_f32_, CV_32FC4);
    d_warp_r_.convertTo(d_r_f32_, CV_32FC4);
    d_warp_l_.convertTo(d_l_f32_, CV_32FC4);
    d_warp_ri_.convertTo(d_ri_f32_, CV_32FC4);

    cv::cuda::multiply(d_f_f32_, d_blend_mask_f_, d_blend_f_);
    cv::cuda::multiply(d_r_f32_, d_blend_mask_r_, d_blend_r_);
    cv::cuda::multiply(d_l_f32_, d_blend_mask_l_, d_blend_l_);
    cv::cuda::multiply(d_ri_f32_, d_blend_mask_ri_, d_blend_ri_);

    if (d_canvas_f32_.empty()) {
        d_canvas_f32_.create(g_.canvas_h, g_.canvas_w, CV_32FC4);
    }
    d_canvas_f32_.setTo(cv::Scalar(0, 0, 0, 0));
    cv::cuda::add(d_canvas_f32_, d_blend_f_, d_canvas_f32_);
    cv::cuda::add(d_canvas_f32_, d_blend_r_, d_canvas_f32_);
    cv::cuda::add(d_canvas_f32_, d_blend_l_, d_canvas_f32_);
    cv::cuda::add(d_canvas_f32_, d_blend_ri_, d_canvas_f32_);

    d_canvas_f32_.convertTo(d_canvas_, CV_8UC4);
    d_canvas_.download(canvas_cpu_4c_);

    // 4. Chuyển BGRx sang BGR (3 kênh) trên CPU
    cv::cvtColor(canvas_cpu_4c_, out_canvas_bgr, cv::COLOR_BGRA2BGR);

    // 5. Chèn logo xe SwiX
    drawPngOverlay(out_canvas_bgr);

    return true;
}

} // namespace camera_stream
