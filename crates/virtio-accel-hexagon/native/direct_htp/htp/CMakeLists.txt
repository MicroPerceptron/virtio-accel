cmake_minimum_required(VERSION 3.22.2)
project(virtio-accel-htp C)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/libs/qhl_hvx/inc/internal
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR})

add_library(virtio-accel-htp-v73 SHARED main.c vector_math.c worker_pool.c va_htp_skel.c)
build_idl(../va_htp.idl virtio-accel-htp-v73)
target_compile_definitions(virtio-accel-htp-v73 PRIVATE NDEBUG=1)
install(TARGETS virtio-accel-htp-v73)
