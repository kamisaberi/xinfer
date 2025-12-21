# src/backends/amd_ryzen_ai/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_ryzen_ai")

# 1. Check for ONNX Runtime (Required for Vitis AI EP)
find_package(OnnxRuntime QUIET)

# 2. Check for Ryzen AI / Vitis AI variables
if(NOT DEFINED ENV{RYZEN_AI_ROOT} AND NOT OnnxRuntime_FOUND)
    message(STATUS "[xInfer] Ryzen AI SDK or ONNX Runtime not found. Skipping 'amd_ryzen_ai' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling AMD Ryzen AI Backend.")

# 3. Add Library
add_library(${BACKEND_NAME} OBJECT
        backend.cpp
)

# 4. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${OnnxRuntime_INCLUDE_DIRS}
        # If using Native XRT
        $ENV{XILINX_XRT}/include
)

# 5. Link Dependencies
# Note: Actual linking happens in the main library, but we define deps here.
target_link_libraries(${BACKEND_NAME} PRIVATE
        xinfer_core
        ${OnnxRuntime_LIBRARIES}
        # If using Native XRT
        # -lxrt_core -lxrt_coreutil
)

# 6. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
        -DXINFER_ENABLE_RYZEN_AI
        -DUSE_VITIS_AI_EP # Enable the Execution Provider logic
)