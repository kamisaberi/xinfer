cmake_minimum_required(VERSION 3.28)
project(xinfer-cli)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find our main `xinfer` library, which this tool depends on.
# This assumes you have already built and installed `xinfer`.
find_package(xinfer REQUIRED)



if(NOT xinfer_FOUND)
    message(FATAL_ERROR "Could not find xinfer library. Please build and install it first.")
endif()

# Define the executable for our command-line tool
add_executable(xinfer-cli main.cpp)

# Link the CLI tool against the `xinfer` library.
# CMake will automatically handle all of xinfer's dependencies (TensorRT, LibTorch, etc.)
target_link_libraries(xinfer-cli PRIVATE xinfer::xinfer)

# Add the path to the cxxopts header
target_include_directories(xinfer-cli PRIVATE ../../third_party)

# Installation rule for the command-line tool
include(GNUInstallDirs)
install(TARGETS xinfer-cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)