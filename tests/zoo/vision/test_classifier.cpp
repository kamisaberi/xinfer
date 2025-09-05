#include <gtest/gtest.h>
#include <xinfer/zoo/vision/classifier.h>
#include <opencv2/opencv.hpp>
#include <fstream>

// This test requires a pre-built engine file and assets in tests/assets/
class ClassifierZooTest : public ::testing::Test {
protected:
    // Path to test assets
    const std::string engine_path = "assets/dummy_resnet18.engine";
    const std::string labels_path = "assets/dummy_labels.txt";
    const std::string image_path = "assets/dummy_image.jpg";

    void SetUp() override {
        // Create dummy assets if they don't exist
        if (!std::ifstream(engine_path).good()) {
            std::cerr << "Warning: Dummy engine not found. This test requires a pre-built engine at "
                      << engine_path << std::endl;
            // In a real CI, a script would build this engine before running tests.
        }
    }
};

TEST_F(ClassifierZooTest, Initialization) {
    xinfer::zoo::vision::ClassifierConfig config;
    config.engine_path = engine_path;
    config.labels_path = labels_path;

    // The test passes if the constructor does not throw an exception
    ASSERT_NO_THROW({
        xinfer::zoo::vision::ImageClassifier classifier(config);
    });
}

TEST_F(ClassifierZooTest, ThrowsOnMissingFile) {
    xinfer::zoo::vision::ClassifierConfig config;
    config.engine_path = "non_existent_file.engine";

    // We expect the constructor to throw a runtime_error if the file is missing
    ASSERT_THROW(xinfer::zoo::vision::ImageClassifier classifier(config), std::runtime_error);
}

TEST_F(ClassifierZooTest, Predict) {
    xinfer::zoo::vision::ClassifierConfig config;
    config.engine_path = engine_path;
    config.labels_path = labels_path;

    xinfer::zoo::vision::ImageClassifier classifier(config);
    cv::Mat image = cv::imread(image_path);
    ASSERT_FALSE(image.empty());

    std::vector<xinfer::zoo::vision::ClassificationResult> results;

    // The test passes if predict does not throw and returns a non-empty result
    ASSERT_NO_THROW({
        results = classifier.predict(image, 5);
    });

    ASSERT_EQ(results.size(), 5);
    for(const auto& res : results) {
        ASSERT_GE(res.confidence, 0.0f);
        ASSERT_LE(res.confidence, 1.0f);
        ASSERT_FALSE(res.label.empty());
    }
}```

### **How to Run Your Test Suite**

1.  **Build your project** with the new `tests/CMakeLists.txt`.
    ```bash
    cd xinfer/build
    cmake ..
    make -j
    ```
2.  **Execute the test runner:**
    ```bash
    # From the build directory
    ./tests/xinfer_tests
    ```
    You will get a clean, formatted output from Google Test showing which tests passed and which failed.

3.  **Use `ctest` for CI/CD:**
    ```bash
    # From the build directory
    ctest --verbose
    ```
    This is the standard way to run tests in an automated continuous integration environment like GitHub Actions.

This comprehensive test suite will give you immense confidence in your library's stability, prevent regressions, and serve as an excellent set of examples for your users.