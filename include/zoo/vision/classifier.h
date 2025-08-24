#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../../hub/model_info.h"

// Forward declarations for clean header
namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

struct ClassificationResult {
    int class_id;
    float confidence;
    std::string label;
};

/**
 * @struct ClassifierConfig
 * @brief Configuration for the ImageClassifier pipeline.
 *
 * This struct defines all the necessary parameters for both the pre-processing
 * pipeline and the model itself.
 */
struct ClassifierConfig {
    // --- Model Path ---
    // Path to the pre-built, optimized TensorRT .engine file.
    std::string engine_path;
    // Path to a simple text file with class names, one per line.
    std::string labels_path = "";

    // --- Pre-processing Options ---
    int input_width = 224;
    int input_height = 224;
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
};

/**
 * @class ImageClassifier
 * @brief A high-level, hyper-optimized pipeline for image classification.
 *
 * This class loads a pre-built TensorRT engine and provides a simple,
 * one-line predict() function that handles all pre- and post-processing.
 */
class ImageClassifier {
public:
    /**
     * @brief The primary constructor.
     * @param config A ClassifierConfig struct with all necessary settings.
     */
    explicit ImageClassifier(const ClassifierConfig& config);
    /**
         * @brief [HUB CONSTRUCTOR] Downloads a pre-built engine and creates a classifier.
         * This is the easiest way to get started.
         * @param model_id The model ID from the Ignition Hub (e.g., "resnet18-imagenet").
         * @param target The specific hardware target to download the engine for.
         */
    explicit ImageClassifier(const std::string& model_id, const xinfer::hub::HardwareTarget& target);

    ~ImageClassifier();

    // Rule of Five for proper resource management
    ImageClassifier(const ImageClassifier&) = delete;
    ImageClassifier& operator=(const ImageClassifier&) = delete;
    ImageClassifier(ImageClassifier&&) noexcept;
    ImageClassifier& operator=(ImageClassifier&&) noexcept;

    /**
     * @brief Runs inference on a single image.
     */
    std::vector<ClassificationResult> predict(const cv::Mat& image, int top_k = 5);

private:
    // PIMPL idiom to hide implementation details
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace xinfer::zoo::vision

