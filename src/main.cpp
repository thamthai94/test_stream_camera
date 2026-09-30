/**
 * *****************************************************************************
 * @file      main.cpp
 * @author    Tapbot Development Team
 * @date      September 2026
 * @brief     Điểm vào (main entry point) của camera_stream_node.
 * *****************************************************************************
 */
#include <memory>
#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>

#include "camera_stream/camera_stream_node.hpp"

int main(int argc, char** argv) {
    // Thiết lập cơ chế đồng bộ CUDA: Blocking Sync thay vì Busy-Spin
    // Tránh việc CPU chạy 100% spin-wait trong khi GPU đang suy luận TensorRT
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    rclcpp::init(argc, argv);
    auto node = std::make_shared<camera_stream::CameraStreamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
