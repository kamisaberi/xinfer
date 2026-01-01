# src/serving/CMakeLists.txt

file(GLOB SERVING_SRCS "*.cpp")

add_library(xinfer_serving OBJECT ${SERVING_SRCS})

# Include third_party for json.hpp and httplib.h
target_include_directories(xinfer_serving PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(xinfer_serving PRIVATE
        xinfer_core
        xinfer_backends
        Threads::Threads # Required by httplib
)

# OpenSSL support (optional, for HTTPS)
find_package(OpenSSL)
if(OPENSSL_FOUND)
    target_link_libraries(xinfer_serving PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(xinfer_serving PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()