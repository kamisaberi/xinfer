# src/compiler/CMakeLists.txt

# Create the master compiler library
add_library(xinfer_compiler OBJECT)

# --- Include All Subdirectories ---
# Drivers are small and don't have heavy dependencies (they shell out),
# so we can build them all unconditionally.
add_subdirectory(drivers)

# Add sources from the drivers library and other local files
target_sources(xinfer_compiler PRIVATE
    $<TARGET_OBJECTS:xinfer_compiler_drivers>
    base_compiler.cpp
    # Add optimizer/quantization files here if you create them
    # optimizer/onnx_surgeon.cpp
    # quantization/calibrator.cpp
)

# --- Link Dependencies ---
target_link_libraries(xinfer_compiler PRIVATE
    xinfer_core
    xinfer_utils # For downloader
)

# --- Expose Headers ---
target_include_directories(xinfer_compiler PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# In the 'drivers' subdirectory...
# src/compiler/drivers/CMakeLists.txt
# ------------------------------------
file(GLOB DRIVER_SOURCES "*.cpp")
add_library(xinfer_compiler_drivers OBJECT ${DRIVER_SOURCES})

target_include_directories(xinfer_compiler_drivers PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)