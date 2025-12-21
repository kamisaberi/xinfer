# src/backends/amd_vitis/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_vitis")

# 1. Check for Vitis AI Runtime (VART) libraries
find_library(LIB_XIR NAMES xir HINTS /usr/lib /usr/local/lib)
find_library(LIB_VART NAMES vart-runner HINTS /usr/lib /usr/local/lib)
find_library(LIB_UNILOG NAMES unilog HINTS /usr/lib /usr/local/lib) # Vitis logging dep

if(NOT LIB_XIR OR NOT LIB_VART)
    message(STATUS "[xInfer] Vitis AI Runtime (VART/XIR) not found. Skipping 'amd_vitis' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling AMD Vitis AI Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
        backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        /usr/include/xir   # Standard location on Kria/Zynq
        /usr/include/vart
)

# 4. Link Dependencies
# We link these to the main library later, but define the requirement here
target_link_libraries(${BACKEND_NAME} PRIVATE
        xinfer_core
        ${LIB_XIR}
        ${LIB_VART}
        ${LIB_UNILOG}
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
        -DXINFER_ENABLE_VITIS
)