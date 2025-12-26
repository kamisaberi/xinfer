#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief A single classification result.
     */
    struct ClassificationResult {
        int id;             // Class Index (e.g., 281)
        float confidence;   // Probability (0.0 - 1.0)
        std::string label;  // Class Name (e.g., "Tabby Cat")
    };

    struct ClassifierConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (.xml, .engine, .rknn, etc.)
        std::string model_path;

        // Path to labels text file (newline separated)
        std::string labels_path;

        // Input Specs
        int input_width = 224;
        int input_height = 224;

        // Normalization (Defaults to ImageNet standards)
        // (Pixel - Mean) / Std
        std::vector<float> mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
        std::vector<float> std  = {0.229f * 255, 0.224f * 255, 0.225f * 255};

        // Post-processing
        int top_k = 1;              // How many results to return
        bool apply_softmax = true;  // Set false if model already includes Softmax layer

        // Vendor flags (e.g. "CORE=0" for Rockchip)
        std::vector<std::string> vendor_params;
    };

    class ImageClassifier {
    public:
        explicit ImageClassifier(const ClassifierConfig& config);
        ~ImageClassifier();

        // Move semantics
        ImageClassifier(ImageClassifier&&) noexcept;
        ImageClassifier& operator=(ImageClassifier&&) noexcept;
        ImageClassifier(const ImageClassifier&) = delete;
        ImageClassifier& operator=(const ImageClassifier&) = delete;

        /**
         * @brief Classify an image.
         *
         * Pipeline:
         * 1. Preprocess (Resize -> Mean/Std Norm -> HWC to NCHW)
         * 2. Inference
         * 3. Postprocess (Softmax -> TopK)
         *
         * @return Vector of top_k results sorted by confidence.
         */
        std::vector<ClassificationResult> classify(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision