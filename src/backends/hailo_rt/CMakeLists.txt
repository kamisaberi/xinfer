# src/backends/hailo_rt/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_hailo")

# 1. Find HailoRT Library
find_library(LIB_HAILORT NAMES hailort HINTS /usr/lib /usr/local/lib)

if(NOT LIB_HAILORT)
    message(STATUS "[xInfer] libhailort not found. Skipping 'hailo_rt' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling HailoRT Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    /usr/include/hailort
)

# 4. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    ${LIB_HAILORT}
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_HAILO
)