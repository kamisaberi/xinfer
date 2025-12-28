cmake_minimum_required(VERSION 3.20)
project(xInfer_Examples)

# --- Find Dependencies ---
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# --- Locate the xInfer Library ---
# Assume xInfer was built in a 'build' directory at the root
# You can also use find_package(xInfer) if you install the library
link_directories(${CMAKE_SOURCE_DIR}/../build)
include_directories(${CMAKE_SOURCE_DIR}/../include)

# --- Define the Example Executable ---
add_executable(wildlife_tracker 01_computer_vision/wildlife_tracker_demo.cpp)

# --- Link Everything Together ---
# This links against the libraries we built (zoo, postproc, etc.)
# and external dependencies like OpenCV and CUDA.
target_link_libraries(wildlife_tracker PRIVATE
        # xInfer Components
        xinfer_zoo
        xinfer_postproc
        xinfer_preproc
        xinfer_backends
        xinfer_core

        # External
        ${OpenCV_LIBS}
        Threads::Threads

        # Conditionally link platform SDKs if they were enabled in the main build
        # These are INTERFACE libraries that bring in CUDA, VART, etc.
        $<$<TARGET_PROPERTY:xinfer_core,XINFER_ENABLE_TRT>:xinfer_cuda>
        $<$<TARGET_PROPERTY:xinfer_core,XINFER_ENABLE_VITIS>:xinfer_vitis>
        $<$<TARGET_PROPERTY:xinfer_core,XINFER_ENABLE_RKNN>:xinfer_rknn>
        # ... and so on for all platforms
)