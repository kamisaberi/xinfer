# src/backends/google_tpu/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_edgetpu")

# 1. Find TensorFlow Lite
find_package(TensorFlowLite QUIET)
find_library(LIB_EDGETPU names edgetpu HINTS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)

if(NOT TensorFlowLite_FOUND AND NOT EXISTS "/usr/include/tensorflow/lite")
    message(STATUS "[xInfer] TensorFlow Lite headers not found. Skipping 'google_tpu' backend.")
    return()
endif()

if(NOT LIB_EDGETPU)
    message(STATUS "[xInfer] libedgetpu not found. Skipping 'google_tpu' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Google Edge TPU Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
        backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        # Standard install locations for TFLite
        /usr/include/tensorflow
        /usr/include/flatbuffers
)

# 4. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
        xinfer_core
        tensorflow-lite
        ${LIB_EDGETPU}
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
        -DXINFER_ENABLE_EDGETPU
)