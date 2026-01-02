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
        xinfer_flow

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

# Telemetry Examples
if(XINFER_BUILD_TELEMETRY)
    add_executable(system_health 09_telemetry/system_health.cpp)
    target_link_libraries(system_health PRIVATE xinfer_core xinfer_telemetry)

    add_executable(detect_drift 09_telemetry/detect_data_drift.cpp)
    target_link_libraries(detect_drift PRIVATE xinfer_core xinfer_telemetry)

    # Note: Remote logging requires networking libs (httplib) linked in xinfer_telemetry
    add_executable(remote_logging 09_telemetry/remote_logging.cpp)
    target_link_libraries(remote_logging PRIVATE xinfer_core xinfer_telemetry)
endif()