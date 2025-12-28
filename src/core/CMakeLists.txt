# src/core/CMakeLists.txt

# Automatically find all source files in this directory
file(GLOB CORE_SOURCES "*.cpp")

# Create the Core library
# It's an OBJECT library so it can be "absorbed" into other libraries later
add_library(xinfer_core OBJECT ${CORE_SOURCES})

# Expose its public headers to any target that links against it
target_include_directories(xinfer_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)