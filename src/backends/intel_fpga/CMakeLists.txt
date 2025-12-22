# src/backends/intel_fpga/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_intel_fpga")

# 1. Check for Intel FPGA Runtime (OpenCL)
# Usually found in /opt/intel/oneapi or /opt/intel/fpgarte
find_package(OpenCL QUIET)

# Check environment variable for DLA Runtime path
if(NOT DEFINED ENV{INTEL_FPGA_AI_SUITE} AND NOT OpenCL_FOUND)
    message(STATUS "[xInfer] Intel FPGA AI Suite or OpenCL not found. Skipping 'intel_fpga' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Intel FPGA Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS}
    $ENV{INTEL_FPGA_AI_SUITE}/include
)

# 4. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    ${OpenCL_LIBRARIES}
    # Link against DLA Runtime if available
    # -L$ENV{INTEL_FPGA_AI_SUITE}/lib -ldla_runtime
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_INTEL_FPGA
)