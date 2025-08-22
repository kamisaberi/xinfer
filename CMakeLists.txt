cmake_minimum_required(VERSION 3.28)
project(xinfer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

#find_package(CUDA REQUIRED)
list(APPEND CMAKE_PREFIX_PATH  /home/kami/libs/cpp/TensorRT-10.13.2.6/)
# Find TensorRT
# You might need to adjust this path to where your TensorRT is installed
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS /usr/include/x86_64-linux-gnu/
)
find_library(TENSORRT_LIBRARY nvinfer
        HINTS /usr/lib/x86_64-linux-gnu/
)
find_library(TENSORRT_PARSER_LIBRARY nvinfer_plugin
        HINTS /usr/lib/x86_64-linux-gnu/
)
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
        HINTS /usr/lib/x86_64-linux-gnu/
)

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT or check paths.")
endif()

message(STATUS "Found TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)



find_package(xTorch REQUIRED)
if(xTorch_FOUND)
    message(STATUS "xTorch config directory found at: ${xTorch_DIR}")

    get_filename_component(XTORCH_INSTALL_DIR "${xTorch_DIR}" DIRECTORY) # up to /usr/local/lib/cmake
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY) # up to /usr/local/lib
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY) # up to /usr/local

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
#find_package(LibZip REQUIRED)

#find_package(onnxruntime REQUIRED)


#set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} ${TRANSFORM_FILES})

set(TEST_LIBRARIES ${CURL_LIBRARIES}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${XTORCH_LIBRARIES}
        xTorch::xTorch
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
       ${BUNDLED_ONNXRUNTIME_LIB_DIR}
        #        onnxruntime
        ONNX::Runtime
        ${LIBZIP_LIBRARIES}  # Add this line
        #        LibZip::zip

        #        LibZip::zip

)

include_directories(SYSTEM ${CMAKE_SOURCE_DIR} /usr/include /usr/local/include ${TENSORRT_INCLUDE_DIR} ${BUNDLED_ONNXRUNTIME_INCLUDE_DIR})

add_executable(xinfer main.cpp)
target_link_libraries(xinfer PRIVATE xTorch::xTorch )
set_property(TARGET xinfer PROPERTY CXX_STANDARD 17)
target_include_directories(xinfer PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})

