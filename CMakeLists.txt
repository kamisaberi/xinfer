cmake_minimum_required(VERSION 3.28)
# THE FIX: Add the VERSION number here. This is required for the install rules.
project(xinfer VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)


# ===================================================================
# 1. FIND DEPENDENCIES (Your original code, unchanged)
# ===================================================================
list(APPEND CMAKE_PREFIX_PATH  /home/kami/libs/cpp/TensorRT-10.13.2.6/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/x86_64-linux-gnu/)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu/)
find_library(TENSORRT_PARSER_LIBRARY nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu/)
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser HINTS /usr/lib/x86_64-linux-gnu/)
if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT or check paths.")
endif()
message(STATUS "Found TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")

find_package(xTorch REQUIRED)
if(xTorch_FOUND)
    message(STATUS "xTorch config directory found at: ${xTorch_DIR}")
    get_filename_component(XTORCH_INSTALL_DIR "${xTorch_DIR}" DIRECTORY)
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY)
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY)
    message(STATUS "Deduced xTorch installation root: ${XTORCH_INSTALL_DIR}")
    set(BUNDLED_LIBTORCH_PATH "${XTORCH_INSTALL_DIR}/include/xtorch/third_party/libtorch")
    if(EXISTS "${BUNDLED_LIBTORCH_PATH}")
        message(STATUS "Found bundled libtorch at: ${BUNDLED_LIBTORCH_PATH}")
        list(PREPEND CMAKE_PREFIX_PATH ${BUNDLED_LIBTORCH_PATH})
    else()
        message(FATAL_ERROR "Could not find the bundled libtorch directory at the expected path: ${BUNDLED_LIBTORCH_PATH}")
    endif()
    set(BUNDLED_ONNXRUNTIME_PATH "${XTORCH_INSTALL_DIR}/include/xtorch/third_party/onnxruntime")
    set(BUNDLED_ONNXRUNTIME_INCLUDE_DIR "${BUNDLED_ONNXRUNTIME_PATH}/include")
    set(BUNDLED_ONNXRUNTIME_LIB_DIR "${BUNDLED_ONNXRUNTIME_PATH}/lib")
endif()
find_package(Torch REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SndFile REQUIRED)

include_directories(SYSTEM ${CMAKE_SOURCE_DIR} /usr/include /usr/local/include ${TENSORRT_INCLUDE_DIR} ${BUNDLED_ONNXRUNTIME_INCLUDE_DIR})


# ===================================================================
# 2. DEFINE THE `xinfer` LIBRARY
# ===================================================================

# Automatically find all library source files from the src/ directory
file(GLOB_RECURSE XINFER_SOURCES "src/*.cpp" "src/*.cu")

# Define `xinfer` as a SHARED library from these source files
add_library(xinfer SHARED ${XINFER_SOURCES})

# Set properties for the library
set_property(TARGET xinfer PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
set_property(TARGET xinfer PROPERTY CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
set_property(TARGET xinfer PROPERTY POSITION_INDEPENDENT_CODE ON)

# Tell the library where to find its own public headers and the headers of its dependencies
target_include_directories(xinfer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${TENSORRT_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${xTorch_INCLUDE_DIRS}
        # Add other include dirs from your TEST_LIBRARIES if needed
)

# Link the library to all of its dependencies
target_link_libraries(xinfer PUBLIC
        xTorch::xTorch
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${TENSORRT_LIBRARY}
        ${TENSORRT_PARSER_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
        # Add other libraries like OpenSSL::SSL etc. if needed
)


# ===================================================================
# 3. DEFINE THE `xinfer_example` EXECUTABLE
# ===================================================================

# This target is built from main.cpp only
add_executable(xinfer_example main.cpp)

# Link the example against your new `xinfer` library.
# CMake handles all the includes and library paths automatically.
target_link_libraries(xinfer_example PRIVATE xinfer)


# ===================================================================
# 4. INSTALLATION RULES (As requested)
# ===================================================================

include(GNUInstallDirs)

# Rule to install the library target
install(TARGETS xinfer
        EXPORT xinfer-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Rule to install the public headers
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Rule to generate and install the CMake config files for `find_package(xinfer)`
install(EXPORT xinfer-targets
        FILE xinfer-config.cmake
        NAMESPACE xinfer::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xinfer
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/xinfer-config-version.cmake"
        VERSION ${PROJECT_VERSION} # This now works because of the fix in the project() line
        COMPATIBILITY AnyNewerVersion
)
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/xinfer-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xinfer
)