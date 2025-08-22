#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declare internal implementation classes to avoid including their headers here.
// This is a key part of the PIMPL idiom and good API design.
namespace xinfer::core { class InferenceEngine; class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision
{
    /**
     * @struct ClassificationResult
     * @brief A simple, human-readable struct to hold a single classification prediction.
     */
    struct ClassificationResult {
        int class_id;
        float confidence;
        std::string label;
    };

    /**
     * @struct ClassifierConfig
     * @brief A configuration structure for creating an ImageClassifier.
     */
    struct ClassifierConfig {
        std::string xtorch_model_path;
        std::string labels_path = "";
        std::string architecture = "ResNet18";
        bool use_fp16 = true;
        bool use_int8 = false;
        std::string int8_calibration_dataset_path = "";
        int max_batch_size = 1;
        size_t workspace_mb = 1024;
        int resize_width = 224;
        int resize_height = 224;
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};
    };

    /**
     * @class ImageClassifier
     * @brief A high-level, hyper-optimized pipeline for image classification.
     */
    class ImageClassifier {
    public:
        /**
         * @brief Advanced constructor. Creates a classifier with a detailed configuration.
         */
        explicit ImageClassifier(const ClassifierConfig& config);

        /**
         * @brief Simple constructor for convenience. Creates a classifier with default settings.
         */
        explicit ImageClassifier(const std::string& xtorch_model_path, const std::string& labels_path = "");

        // Destructor must be declared because of the unique_ptr to an incomplete type (PIMPL).
        ~ImageClassifier();

        // Rule of Five for proper resource management with PIMPL
        ImageClassifier(const ImageClassifier&) = delete;
        ImageClassifier& operator=(const ImageClassifier&) = delete;
        ImageClassifier(ImageClassifier&&) noexcept;
        ImageClassifier& operator=(ImageClassifier&&) noexcept;

        /**
         * @brief Runs inference on a single image provided as an OpenCV Mat.
         */
        std::vector<ClassificationResult> predict(const cv::Mat& image, int top_k = 5);

    private:
        // PIMPL (Pointer to Implementation) idiom.
        // This hides all the complex member variables (engine, preprocessor, etc.)
        // from the user of the header file, reducing compile-time dependencies.
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };
}