# src/preproc/CMakeLists.txt

# --- 1. Universal Fallback: OpenCV ---
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- 2. NVIDIA CUDA ---
if(XINFER_ENABLE_TRT)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(XINFER_HAS_CUDA)
endif()

# --- 3. Rockchip RGA (Hardware 2D) ---
find_library(LIB_RGA names rga HINTS /usr/lib /usr/lib/aarch64-linux-gnu)
if(LIB_RGA)
    message(STATUS "[xInfer] Found Rockchip librga. Enabling Hardware Preprocessing.")
    add_compile_definitions(XINFER_HAS_RGA)
    include_directories(/usr/include/rga) # Standard RGA header path
endif()

# --- 4. Define Sources ---
set(PREPROC_SRCS
        factory.cpp
        image/cpu/opencv_image.cpp
        audio/cpu/cpu_audio.cpp
)

if(XINFER_ENABLE_TRT)
    list(APPEND PREPROC_SRCS image/cuda/cuda_image.cu)
endif()

if(LIB_RGA)
    list(APPEND PREPROC_SRCS image/rga/rga_image.cpp)
endif()

# --- 5. Create Library ---
add_library(xinfer_preproc OBJECT ${PREPROC_SRCS})

target_include_directories(xinfer_preproc PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies to the main library later, or use interface logic here