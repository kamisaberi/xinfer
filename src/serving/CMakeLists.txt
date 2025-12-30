# src/serving/CMakeLists.txt

# Find sources
file(GLOB_RECURSE SERVING_SRCS "*.cpp")

# Create Library
add_library(xinfer_serving OBJECT ${SERVING_SRCS})

# Dependencies
target_link_libraries(xinfer_serving PRIVATE
        xinfer_core
        xinfer_backends
        Threads::Threads # Required for cpp-httplib
)

# Include Paths
target_include_directories(xinfer_serving PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party # For httplib.h, json.hpp
)

# Compiler Flags
if(UNIX)
    # httplib recommends this for performance
    target_compile_definitions(xinfer_serving PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()