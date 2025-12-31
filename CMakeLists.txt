cmake_minimum_required(VERSION 3.20)
project(xInfer LANGUAGES CXX CUDA C OBJCXX) # Enable languages for all platforms

# ==============================================================================
# 1. Project-wide Settings
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position-independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================================================================
# 2. Build Options (User can toggle these with -D)
# ==============================================================================
option(XINFER_ENABLE_TRT "Enable NVIDIA TensorRT Backend" OFF)
option(XINFER_ENABLE_VITIS "Enable AMD/Xilinx Vitis Backend" OFF)
option(XINFER_ENABLE_RKNN "Enable Rockchip RKNN Backend" OFF)
option(XINFER_ENABLE_QNN "Enable Qualcomm QNN Backend" OFF)
option(XINFER_ENABLE_OPENVINO "Enable Intel OpenVINO Backend" ON)
option(XINFER_ENABLE_COREML "Enable Apple CoreML Backend" ${APPLE})
# ... Add options for all 15 backends ...

option(XINFER_BUILD_EXAMPLES "Build the example applications" ON)
option(XINFER_BUILD_TOOLS "Build the xinfer-cli tool" ON)


option(XINFER_BUILD_SERVING   "Build HTTP/REST Model Server"  ON)
option(XINFER_BUILD_FLOW      "Build Pipeline Orchestrator"   ON)
option(XINFER_BUILD_TELEMETRY "Build Observability System"    ON)


# ==============================================================================
# 3. Find External Dependencies
# ==============================================================================
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED) # For the downloader

if(XINFER_ENABLE_TRT)
    find_package(CUDAToolkit REQUIRED)
    # Logic to find TensorRT would go here, often via environment variables
    # (Simplified for this example)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ==============================================================================
# 4. Include Source Directories
# ==============================================================================
# This is where we build our modular components

# --- Core Library ---
# The foundation: Tensor, Logging, DeviceManager
add_subdirectory(src/core)

# --- Utilities ---
add_subdirectory(src/utils) # For Downloader

# --- Backends ---
# This directory will contain CMake files for each of the 15 runtime backends
add_subdirectory(src/backends)

# --- Preprocessing ---
# This builds the preproc library with conditional CUDA/RGA/NEON support
add_subdirectory(src/preproc)

# --- Post-processing ---
# This builds the postproc library with conditional CUDA/CPU support
add_subdirectory(src/postproc)

# --- Compiler ---
# This builds the compiler driver library
add_subdirectory(src/compiler)

# --- Model Zoo ---
# This builds the high-level application modules
add_subdirectory(src/zoo)

# ==============================================================================
# 5. Build Tools & Examples (if enabled)
# ==============================================================================

if(XINFER_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(XINFER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(XINFER_BUILD_SERVING)
    message(STATUS "[xInfer] Enabling Serving Module")
    add_subdirectory(src/serving)
endif()

if(XINFER_BUILD_FLOW)
    message(STATUS "[xInfer] Enabling Flow Module")
    add_subdirectory(src/flow)
endif()

if(XINFER_BUILD_TELEMETRY)
    message(STATUS "[xInfer] Enabling Telemetry Module")
    add_subdirectory(src/telemetry)
endif()

# ==============================================================================
# 6. Final Library Aggregation (Optional but good practice)
# ==============================================================================
# You can create a single 'xinfer' INTERFACE library that bundles all public
# headers and links all components for easy consumption by other projects.

add_library(xinfer INTERFACE)
target_include_directories(xinfer INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(xinfer INTERFACE
    xinfer_core
    xinfer_utils
    xinfer_backends # This is an object library that needs to be linked into a final target
    xinfer_preproc
    xinfer_postproc
    xinfer_zoo
    
    # Propagate public dependencies
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    Threads::Threads
)


# ==============================================================================
# Post-Build: Copy Configs
# ==============================================================================
# This copies the entire 'configs' folder into 'build/configs' automatically
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/configs
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})







message(STATUS "[xInfer] Config files copied to: ${CMAKE_CURRENT_BINARY_DIR}/configs")