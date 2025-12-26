#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    struct ImageSimilarityConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., resnet50_feature_extractor.onnx)
        // Should output a flattened feature vector (e.g. 1x2048)
        std::string model_path;

        // Input Specs
        int input_width = 224;
        int input_height = 224;

        // Normalization (Standard ImageNet defaults)
        std::vector<float> mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
        std::vector<float> std  = {0.229f * 255, 0.224f * 255, 0.225f * 255};

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ImageSimilarity {
    public:
        explicit ImageSimilarity(const ImageSimilarityConfig& config);
        ~ImageSimilarity();

        // Move semantics
        ImageSimilarity(ImageSimilarity&&) noexcept;
        ImageSimilarity& operator=(ImageSimilarity&&) noexcept;
        ImageSimilarity(const ImageSimilarity&) = delete;
        ImageSimilarity& operator=(const ImageSimilarity&) = delete;

        /**
         * @brief Compute similarity between two images.
         *
         * @param img1 First image (BGR).
         * @param img2 Second image (BGR).
         * @return Cosine Similarity score [-1.0, 1.0].
         *         1.0 = Identical, 0.0 = Unrelated.
         */
        float compare(const cv::Mat& img1, const cv::Mat& img2);

        /**
         * @brief Extract feature vector for a single image.
         * Useful for building a search database.
         *
         * @return Normalized feature vector.
         */
        std::vector<float> get_features(const cv::Mat& img);

        /**
         * @brief Calculate similarity between two pre-computed feature vectors.
         */
        static float compute_cosine_similarity(const std::vector<float>& vec_a,
                                               const std::vector<float>& vec_b);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision