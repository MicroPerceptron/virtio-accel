if(HEXAGON_TOOLCHAIN_INCLUDED)
  return()
endif()
set(HEXAGON_TOOLCHAIN_INCLUDED true)
set(HEXAGON TRUE)
set(CMAKE_SYSTEM_NAME QURT)
set(CMAKE_SYSTEM_PROCESSOR Hexagon)
set(CMAKE_SYSTEM_VERSION "1")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

file(TO_CMAKE_PATH "${HEXAGON_SDK_ROOT}" HEXAGON_SDK_ROOT)
file(TO_CMAKE_PATH "${HEXAGON_TOOLS_ROOT}" HEXAGON_TOOLS_ROOT)
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT DSP_VERSION PREBUILT_LIB_DIR)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
  set(HEXAGON_TOOLCHAIN_SUFFIX .exe)
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_arch.cmake)
set(HEXAGON_TOOLCHAIN ${HEXAGON_TOOLS_ROOT})
set(HEXAGON_LIB_DIR "${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib")
set(V_ARCH ${HEXAGON_ARCH})
set(RTOS_DIR "${HEXAGON_SDK_ROOT}/rtos/qurt/compute${V_ARCH}${V_ARCH_EXTN}")
set(TARGET_DIR "${HEXAGON_LIB_DIR}/${V_ARCH}/G0")
set(TARGET_DIR_NOOS "${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib/${HEXAGON_ARCH}")

include_directories(SYSTEM
  ${HEXAGON_SDK_ROOT}/incs
  ${HEXAGON_SDK_ROOT}/incs/stddef
  ${HEXAGON_SDK_ROOT}/ipc/fastrpc/incs
  ${RTOS_DIR}/include
  ${RTOS_DIR}/include/qurt
  ${RTOS_DIR}/include/posix)

set(CMAKE_C_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_AR ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-ar${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Wl,-soname,")
set(ARCH_FLAGS "-mcpu=${V_ARCH} -m${V_ARCH} -mhvx=${V_ARCH} -mhmx")
set(COMMON_FLAGS "${ARCH_FLAGS} -fvectorize -flto -Wall -Werror -fno-zero-initialized-in-bss -G0 -fdata-sections -fpic")
set(CMAKE_C_FLAGS_RELEASE "${COMMON_FLAGS} -O3")

set(CMAKE_C_CREATE_SHARED_LIBRARY
  "${CMAKE_C_COMPILER} ${ARCH_FLAGS} -G0 -fpic -Wl,-Bsymbolic -Wl,-L${TARGET_DIR_NOOS}/G0/pic -Wl,-L${HEXAGON_LIB_DIR} -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -shared -o <TARGET> <SONAME_FLAG><TARGET_SONAME> <LINK_FLAGS> -Wl,--start-group <OBJECTS> <LINK_LIBRARIES> -Wl,--end-group -lc")
