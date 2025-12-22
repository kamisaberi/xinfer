# src/backends/lattice_sensai/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_lattice")

# 1. Find FTDI Library (for USB communication)
find_package(PkgConfig QUIET)
pkg_check_modules(FTDI libftdi1)

if(NOT FTDI_FOUND)
    message(STATUS "[xInfer] libftdi1 not found. Lattice backend will support SPI/File mode only.")
    # We don't return here, we just disable USB features via a define
else()
    message(STATUS "[xInfer] Enabling Lattice Backend with USB/FTDI support.")
    add_compile_definitions(XINFER_ENABLE_FTDI)
endif()

message(STATUS "[xInfer] Enabling Lattice sensAI Backend.")

# 2. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 3. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${FTDI_INCLUDE_DIRS}
)

# 4. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    ${FTDI_LIBRARIES}
)

# 5. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_LATTICE
)