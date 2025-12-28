# src/backends/CMakeLists.txt

# --- Create a "master" object library for all enabled backends ---
add_library(xinfer_backends OBJECT)

# --- Conditionally Include Backends ---

# NVIDIA TensorRT
if(XINFER_ENABLE_TRT)
    message(STATUS "[xInfer] Building Backend: NVIDIA TensorRT")
    add_subdirectory(nvidia_trt)
    # Add the sources from the nvidia_trt library to our master library
    target_sources(xinfer_backends PRIVATE
        $<TARGET_OBJECTS:xinfer_backend_nvidia>
    )
endif()

# Intel OpenVINO
if(XINFER_ENABLE_OPENVINO)
    message(STATUS "[xInfer] Building Backend: Intel OpenVINO")
    add_subdirectory(intel_ov)
    target_sources(xinfer_backends PRIVATE
        $<TARGET_OBJECTS:xinfer_backend_openvino>
    )
endif()

# Rockchip RKNN
if(XINFER_ENABLE_RKNN)
    message(STATUS "[xInfer] Building Backend: Rockchip RKNN")
    add_subdirectory(rockchip_rknn)
    target_sources(xinfer_backends PRIVATE
        $<TARGET_OBJECTS:xinfer_backend_rknn>
    )
endif()

# Apple CoreML
if(XINFER_ENABLE_COREML)
    message(STATUS "[xInfer] Building Backend: Apple CoreML")
    add_subdirectory(apple_coreml)
    target_sources(xinfer_backends PRIVATE
        $<TARGET_OBJECTS:xinfer_backend_coreml>
    )
endif()

# ... (Add an 'if' block for all 15 backends) ...
# Example for Vitis:
# if(XINFER_ENABLE_VITIS)
#   add_subdirectory(amd_vitis)
#   target_sources(xinfer_backends PRIVATE $<TARGET_OBJECTS:xinfer_backend_vitis>)
# endif()


# --- Propagate Includes & Links ---

# The master backends library needs to know about the core library headers
target_link_libraries(xinfer_backends PRIVATE xinfer_core)

# Expose the public include path
target_include_directories(xinfer_backends PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)