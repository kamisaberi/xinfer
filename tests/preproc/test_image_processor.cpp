#include <gtest/gtest.h>
#include <xinfer/xinfer.h>
#include <xinfer/xinfer.h>
#include <opencv2/opencv.hpp>
#include <vector>

class ImageProcessorTest : public ::testing::Test {};

TEST_F(ImageProcessorTest, ResizesAndNormalizesCorrectly) {
    // 1. Configure the pre-processor for standard ImageNet normalization
    int target_h = 224, target_w = 224;
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
    xinfer::preproc::ImageProcessor preprocessor(target_w, target_h, mean, std);

    // 2. Create a dummy input image with a known color
    // OpenCV uses BGR order: Blue=200, Green=100, Red=50
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(200, 100, 50));

    // 3. Create a destination tensor and process the image
    xinfer::core::Tensor gpu_tensor({1, 3, target_h, target_w}, xinfer::core::DataType::kFLOAT);
    preprocessor.process(image, gpu_tensor);

    // 4. Download the result and verify the values
    std::vector<float> h_tensor(gpu_tensor.num_elements());
    gpu_tensor.copy_to_host(h_tensor.data());

    // Calculate the expected normalized values
    float expected_r = (50.0f / 255.0f - mean[0]) / std[0];
    float expected_g = (100.0f / 255.0f - mean[1]) / std[1];
    float expected_b = (200.0f / 255.0f - mean[2]) / std[2];

    // Check a pixel in the center of each channel plane
    int center_idx = (target_h / 2) * target_w + (target_w / 2);
    float r_val = h_tensor[0 * target_h * target_w + center_idx];
    float g_val = h_tensor[1 * target_h * target_w + center_idx];
    float b_val = h_tensor[2 * target_h * target_w + center_idx];

    ASSERT_NEAR(r_val, expected_r, 1e-6);
    ASSERT_NEAR(g_val, expected_g, 1e-6);
    ASSERT_NEAR(b_val, expected_b, 1e-6);
}