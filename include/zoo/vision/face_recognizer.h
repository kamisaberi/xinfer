#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    struct FaceRecognizerConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., arcface_mobile.rknn)
        std::string model_path;

        // Input Specs (ArcFace standard is 112x112)
        int input_width = 112;
        int input_height = 112;

        // Normalization (Standard ArcFace: Mean=127.5, Std=128.0)
        std::vector<float> mean = {127.5f, 127.5f, 127.5f};
        std::vector<float> std  = {128.0f, 128.0f, 128.0f};

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class FaceRecognizer {
    public:
        explicit FaceRecognizer(const FaceRecognizerConfig& config);
        ~FaceRecognizer();

        // Move semantics
        FaceRecognizer(FaceRecognizer&&) noexcept;
        FaceRecognizer& operator=(FaceRecognizer&&) noexcept;
        FaceRecognizer(const FaceRecognizer&) = delete;
        FaceRecognizer& operator=(const FaceRecognizer&) = delete;

        /**
         * @brief Extract feature embedding from a face crop.
         *
         * @param face_image Input image (BGR), typically aligned/cropped.
         * @return Normalized feature vector (usually 512 floats).
         */
        std::vector<float> get_embedding(const cv::Mat& face_image);

        /**
         * @brief Calculate Cosine Similarity between two embeddings.
         *
         * @return Score between -1.0 and 1.0.
         *         Typically > 0.3 or 0.4 indicates a match (depending on model).
         */
        static float compute_similarity(const std::vector<float>& emb1,
                                        const std::vector<float>& emb2);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision