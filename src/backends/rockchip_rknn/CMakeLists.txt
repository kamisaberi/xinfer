# src/backends/rockchip_rknn/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_rknn")

# 1. Find RKNN API (RKNPU2)
# Usually installed in /usr/include/rknn via rknpu2-dev package or manual install
find_path(RKNN_INCLUDE_DIR NAMES rknn_api.h HINTS /usr/include /usr/local/include /usr/include/rknn)
find_library(RKNN_LIB NAMES rknnrt HINTS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib)

if(NOT RKNN_INCLUDE_DIR OR NOT RKNN_LIB)
    message(STATUS "[xInfer] RKNN Runtime (librknnrt) not found. Skipping 'rockchip_rknn' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Rockchip RKNN Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${RKNN_INCLUDE_DIR}
)

# 4. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    ${RKNN_LIB}
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_RKNN
)