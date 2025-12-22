# src/backends/microchip_vb/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_microchip")

# 1. Check for VectorBlox SDK
if(NOT DEFINED ENV{VECTORBLOX_SDK})
    message(STATUS "[xInfer] VECTORBLOX_SDK not set. Skipping 'microchip_vb' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Microchip VectorBlox Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    $ENV{VECTORBLOX_SDK}/inc
    $ENV{VECTORBLOX_SDK}/drivers/vbx_cnn
)

# 4. Link Dependencies
# VectorBlox SDK is often compiled as a static lib (libvbx_cnn.a)
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    $ENV{VECTORBLOX_SDK}/lib/libvbx_cnn.a
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_MICROCHIP
    -DVBX_SOC_POLARFIRE # Or generic RISC-V defs
)