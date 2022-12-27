cmake_minimum_required(VERSION 3.5)
project(yolov5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/root/openvinox/yolov5")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
set(ENV{PKG_CONFIG_PATH} "ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)
set(OpenVINO_DIR /opt/intel/openvino/runtime/cmake/)
find_package(OpenVINO REQUIRED)


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)

list(APPEND PRO_SOURCESE ${SRCS})
add_executable (${PROJECT_NAME} ${PRO_SOURCESE})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
        
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        openvino::runtime)
