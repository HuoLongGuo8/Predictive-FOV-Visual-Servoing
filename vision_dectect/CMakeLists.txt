cmake_minimum_required(VERSION 3.0.2)
project(uav_vision_dectect)

# Default to C++17 (required for ONNX Runtime)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  sensor_msgs
  uav_common_msg
)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(ONNXRUNTIME_ROOT_DIR "/home/fly/onnxruntime" CACHE PATH "ONNX Runtime 安装根目录")

if(NOT EXISTS "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "找不到 onnxruntime_cxx_api.h\n"
    "  期望路径：${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime_cxx_api.h\n"
    "  请确认 ONNXRUNTIME_ROOT_DIR 指向正确的 ONNX Runtime 安装目录，或通过 cmake -D 修改：\n"
    "    catkin_make -DONNXRUNTIME_ROOT_DIR=/home/fly/onnxruntime")
endif()

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
  NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR
    "找不到 libonnxruntime.so\n"
    "  期望路径：${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so\n"
    "  请确认 ONNXRUNTIME_ROOT_DIR 指向正确的 ONNX Runtime 安装目录，或通过 cmake -D 修改：\n"
    "    catkin_make -DONNXRUNTIME_ROOT_DIR=/home/fly/onnxruntime")
endif()

message(STATUS "ONNX Runtime include : ${ONNXRUNTIME_ROOT_DIR}/include")
message(STATUS "ONNX Runtime library : ${ONNXRUNTIME_LIB}")

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs cv_bridge sensor_msgs uav_common_msg
)

include_directories(
  ./include/uav_vision_dectect/
  ./include/light_track
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  "${ONNXRUNTIME_ROOT_DIR}/include"
)

add_executable(uav_vision_dectect
  ./src/main.cpp
  ./src/uav_topic_subscrib.cpp
  ./src/uav_control.cpp
  ./src/LightTrack.cpp
  ./src/yolo_detector.cpp
)

add_dependencies(uav_vision_dectect ${catkin_EXPORTED_TARGETS})

target_link_libraries(uav_vision_dectect
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(uav_vision_dectect
    OpenMP::OpenMP_CXX
  )
endif()

install(TARGETS uav_vision_dectect
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)