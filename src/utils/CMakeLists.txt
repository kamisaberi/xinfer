# src/utils/CMakeLists.txt

# Find Dependencies
find_package(CURL REQUIRED)

# Find Sources
file(GLOB UTILS_SOURCES "*.cpp")

# Create Library
add_library(xinfer_utils OBJECT ${UTILS_SOURCES})

# Link against libcurl
target_link_libraries(xinfer_utils PRIVATE
    ${CURL_LIBRARIES}
)

# Expose public headers and include directories
target_include_directories(xinfer_utils PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_include_directories(xinfer_utils PRIVATE
    ${CURL_INCLUDE_DIRS}
)