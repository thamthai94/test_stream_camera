cmake_minimum_required(VERSION 3.18)
project(camera_stream LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CUDA_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Thiết lập CUDA architecture cho Jetson Orin (Ampere sm_87)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()


set(OpenCV_DIR /opt/opencv_cuda/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# Thiết lập RPATH để binary tìm thấy các thư viện tại /opt/opencv_cuda/lib khi chạy ros2 launch
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "/opt/opencv_cuda/lib;/usr/local/cuda/lib64;/usr/lib/aarch64-linux-gnu/tegra")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)

# Tìm các thư viện CUDA, TensorRT và Jetson Multimedia API (NvBufSurface)
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
find_library(NVBUFSURFACE_LIB nvbufsurface HINTS /usr/lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/tegra)

add_executable(camera_stream_node
  src/mosaic_kernel.cu
  src/face_anonymizer.cpp
  src/plate_detector.cpp
  src/birdview_processor.cpp
  src/birdview_encoder.cpp
  src/camera_stream_node.cpp
  src/main.cpp
)

target_compile_features(camera_stream_node PUBLIC cxx_std_17)
target_include_directories(camera_stream_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GSTREAMER_INCLUDE_DIRS}
  /usr/src/jetson_multimedia_api/include
  /usr/include/aarch64-linux-gnu
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(camera_stream_node
  ${GSTREAMER_LIBRARIES}
  CUDA::cudart
  ${NVINFER_LIB}
  ${NVBUFSURFACE_LIB}
)

target_link_libraries(camera_stream_node 
  ${OpenCV_LIBS}
)

ament_target_dependencies(camera_stream_node
  rclcpp
  sensor_msgs
  std_srvs
  ament_index_cpp
)

set_target_properties(camera_stream_node PROPERTIES
  BUILD_RPATH "/opt/opencv_cuda/lib;/usr/local/cuda/lib64;/usr/lib/aarch64-linux-gnu/tegra"
  INSTALL_RPATH "/opt/opencv_cuda/lib;/usr/local/cuda/lib64;/usr/lib/aarch64-linux-gnu/tegra"
)

install(TARGETS camera_stream_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
