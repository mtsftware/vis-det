cmake_minimum_required(VERSION 3.16)
project(MyApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GStreamer 1.0 gerekli
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)

include_directories(${CMAKE_SOURCE_DIR}/include ${GSTREAMER_INCLUDE_DIRS})
link_directories(${GSTREAMER_LIBRARY_DIRS})

# Ana uygulama
add_executable(myapp
    src/main.cpp
    src/ingestion/RtspDemuxer.cpp
    src/decode/MppDecoder.cpp
)

target_link_libraries(myapp PRIVATE ${GSTREAMER_LDFLAGS})
target_include_directories(myapp PRIVATE ${GSTREAMER_INCLUDE_DIRS})

# RK3588 hedefinde Rockchip MPP kütüphaneleri için tanımlar
# Hedef derlemede şunları eklemek gerekebilir:
#   -lmpp -ldcsm -lrga