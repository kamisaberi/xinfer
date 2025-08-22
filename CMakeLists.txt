cmake_minimum_required(VERSION 3.28)
project(xinfer)

set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)



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
        #        ${ONNXRUNTIME_LIBRARY}
        #        onnxruntime
        ONNX::Runtime
        ${LIBZIP_LIBRARIES}  # Add this line
        #        LibZip::zip

        #        LibZip::zip

)

include_directories(SYSTEM ${CMAKE_SOURCE_DIR} /usr/include /usr/local/include)

add_executable(xinfer main.cpp)
target_link_libraries(xinfer PRIVATE xTorch::xTorch)
set_property(TARGET xinfer PROPERTY CXX_STANDARD 17)
target_include_directories(xinfer PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})

