# src/backends/ambarella_cv/CMakeLists.txt

# 1. Define the backend name
set(BACKEND_NAME "xinfer_backend_ambarella")

# 2. Check if the Proprietary SDK path is set (Environment Variable)
if(NOT DEFINED ENV{AMBA_CVFLOW_SDK})
    message(STATUS "[xInfer] Ambarella SDK not found. Skipping 'ambarella_cv' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling Ambarella CVFlow Backend.")
set(AMBA_SDK_ROOT $ENV{AMBA_CVFLOW_SDK})

# 3. Define the library
add_library(${BACKEND_NAME} OBJECT
        backend.cpp
        # If you have a separate provider/factory registration file:
        # provider.cpp
)

# 4. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        # Proprietary Headers (hidden from the rest of xInfer)
        ${AMBA_SDK_ROOT}/include
        ${AMBA_SDK_ROOT}/include/cavalry
)

# 5. Link Libraries (The .a or .so files from the SDK)
# Note: Since this is an OBJECT library, we usually link these in the main xinfer lib,
# but strictly speaking, we define dependencies here.
target_link_libraries(${BACKEND_NAME} PRIVATE
        xinfer_core
        # Link against the simulated or real Ambarella driver libs
        ${AMBA_SDK_ROOT}/lib/libcalvary.so
        ${AMBA_SDK_ROOT}/lib/libambadag.so
)

# 6. Compiler Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
        -DXINFER_ENABLE_AMBARELLA
        -D_GNU_SOURCE # Required for some low-level Linux ioctl calls
)