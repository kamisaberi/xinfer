# src/backends/samsung_exynos/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_exynos")

# 1. Check for Samsung ENN SDK
# Usually found in Android NDK vendor folders or a standalone BSP
if(NOT DEFINED ENV{ENN_SDK_ROOT})
    message(STATUS "[xInfer] ENN_SDK_ROOT not set. Skipping 'samsung_exynos' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Samsung Exynos Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    $ENV{ENN_SDK_ROOT}/include
)

# 4. Link Dependencies
# Link against the ENN Common API (libenn_common.so / libenn_public_api_cpp.so)
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    $ENV{ENN_SDK_ROOT}/lib/libenn_public_api_cpp.so
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_EXYNOS
    -DENN_medium_precision # Default mode often used
)