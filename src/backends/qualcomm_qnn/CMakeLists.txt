# src/backends/qualcomm_qnn/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_qnn")

# 1. Check for QNN SDK
if(NOT DEFINED ENV{QNN_SDK_ROOT})
    message(STATUS "[xInfer] QNN_SDK_ROOT not set. Skipping 'qualcomm_qnn' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Qualcomm QNN Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    $ENV{QNN_SDK_ROOT}/include/QNN
)

# 4. Link Dependencies
# QNN relies heavily on dlopen (libdl) to load the HTP/GPU backends at runtime.
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    ${CMAKE_DL_LIBS} # -ldl on Linux
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_QNN
)