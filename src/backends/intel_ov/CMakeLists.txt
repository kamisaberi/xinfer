# src/backends/intel_ov/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_openvino")

# 1. Find OpenVINO Package
# Standard installation puts this in system paths or TBB paths
find_package(OpenVINO QUIET)

if(NOT OpenVINO_FOUND)
    message(STATUS "[xInfer] OpenVINO not found. Skipping 'intel_ov' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Intel OpenVINO Backend (Found version: ${OpenVINO_VERSION}).")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

# 4. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    openvino::runtime
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_OPENVINO
)