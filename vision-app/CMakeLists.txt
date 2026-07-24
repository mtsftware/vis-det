cmake_minimum_required(VERSION 3.10)
project(MyApp)

# C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Derleme bayrakları
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

# Release build varsayılan
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ========================
#_include path'leri
# ========================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include/rockchip
    /usr/include/rga
)

# ========================
# Kaynak dosyalari
# ========================
set(SOURCES
    src/main.cpp
    src/buffer/DmaBufferPool.cpp
    src/decode/MppDecoder.cpp
    src/inference/YoloInferenceEngine.cpp
    src/ingestion/RtspDemuxer.cpp
    src/pipeline/PipelineOrchestrator.cpp
    src/postprocess/YoloPostProcessor.cpp
    src/rga_processing/RgaPreprocessor.cpp
    src/streaming/RtspStreamer.cpp
    src/tracking/MultiObjectTracker.cpp
)

# ========================
# Harici kütüphaneler
# ========================
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-rtsp-server-1.0
)

include_directories(${GST_INCLUDE_DIRS})
link_directories(${GST_LIBRARY_DIRS})

# ========================
# RKNN/MPP/RGA
# ========================
set(RKNN_LIB ${CMAKE_CURRENT_SOURCE_DIR}/third_party/librknn_api/aarch64/librknnrt.so)
set(RGA_LIB rga)
set(MPP_LIB rockchip_mpp)
set(PTHREAD pthread)

# ========================
# Çıktı
# ========================
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${RKNN_LIB}
    ${RGA_LIB}
    ${MPP_LIB}
    ${GST_LIBRARY_DIRS}
    ${GST_LDFLAGS}
    ${PTHREAD}
)