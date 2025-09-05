# This tells CMake to enable testing for this project
enable_testing()

# Find the Google Test package.
# This is the standard way to use gtest with CMake.
find_package(GTest REQUIRED)

# ===================================================================
#  Define the main test executable
# ===================================================================
add_executable(xinfer_tests
        main.cpp
        # --- Add every single test file here ---
        core/test_tensor.cpp
        core/test_engine.cpp
        builders/test_engine_builder.cpp
        preproc/test_image_processor.cpp
        postproc/test_detection.cpp
        zoo/vision/test_classifier.cpp
)

# Link the test executable against our main `xinfer` library,
# Google Test, and any other necessary dependencies.
target_link_libraries(xinfer_tests PRIVATE
        xinfer::xinfer  # Link to the main library we are testing
        GTest::gtest_main # This provides the main() function and links gtest
)

# Add the test to CTest so we can run it with `ctest`
include(GoogleTest)
gtest_discover_tests(xinfer_tests)