# src/flow/CMakeLists.txt

# Find sources
file(GLOB_RECURSE FLOW_SRCS "*.cpp")

# Create Library
add_library(xinfer_flow OBJECT ${FLOW_SRCS})

# Dependencies
target_link_libraries(xinfer_flow PRIVATE
        xinfer_core
        xinfer_preproc
        xinfer_postproc
        xinfer_zoo       # Flow wraps Zoo modules
        xinfer_backends  # Direct inference nodes
        ${OpenCV_LIBS}   # For Source/Sink nodes (Camera/Display)
)

# Include Paths
target_include_directories(xinfer_flow PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party # For json.hpp (Config parsing)
)