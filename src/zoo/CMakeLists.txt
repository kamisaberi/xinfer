# src/zoo/CMakeLists.txt

# Automatically find all source files in all subdirectories of zoo/
file(GLOB_RECURSE ZOO_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
)

# Create the Zoo Library
add_library(xinfer_zoo OBJECT ${ZOO_SOURCES})

# --- Link against the Core Components ---
# The Zoo depends on all the lower-level modules.
target_link_libraries(xinfer_zoo PRIVATE
        xinfer_core
        xinfer_preproc
        xinfer_postproc
        xinfer_backends # Assumes a single library for all backends

        # External dependencies
        ${OpenCV_LIBS}
        # ... any other common dependencies
)

# --- Include Directories ---
# Zoo files need to see the public headers of the entire project
target_include_directories(xinfer_zoo PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Conditional Flags ---
# Pass down flags for platform-specific files
if(XINFER_ENABLE_TRT)
    target_compile_definitions(xinfer_zoo PRIVATE XINFER_HAS_CUDA)
endif()

if(APPLE)
    target_compile_options(xinfer_zoo PRIVATE -fobjc-arc)
endif()