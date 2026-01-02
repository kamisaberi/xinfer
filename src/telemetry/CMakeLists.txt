# src/telemetry/CMakeLists.txt

# Automatically find all .cpp files in this directory and subdirectories (exporters/)
file(GLOB_RECURSE TELEMETRY_SRCS "*.cpp")

# Create the Object Library
add_library(xinfer_telemetry OBJECT ${TELEMETRY_SRCS})

# --- Dependencies ---
target_link_libraries(xinfer_telemetry PRIVATE
        xinfer_core
        Threads::Threads
)

# --- Include Directories ---
target_include_directories(xinfer_telemetry PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party # For json.hpp and httplib.h
)

# --- Platform Specifics ---
if(XINFER_ENABLE_TRT)
    # If we want GPU stats via NVML later
    target_compile_definitions(xinfer_telemetry PRIVATE XINFER_HAS_GPU_MONITORING)
endif()