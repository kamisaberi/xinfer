# src/backends/mediatek_np/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_mediatek")

# 1. Check for NeuroPilot SDK (Neuron)
find_path(NEURON_INCLUDE_DIR NAMES neuron/neuron.h HINTS /usr/include /usr/local/include)
find_library(NEURON_LIB NAMES neuronusdk_adapter HINTS /usr/lib /usr/local/lib)

if(NOT NEURON_INCLUDE_DIR OR NOT NEURON_LIB)
    message(STATUS "[xInfer] MediaTek NeuroPilot SDK not found. Skipping 'mediatek_np' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling MediaTek NeuroPilot Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${NEURON_INCLUDE_DIR}
)

# 4. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    ${NEURON_LIB}
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_MEDIATEK
)