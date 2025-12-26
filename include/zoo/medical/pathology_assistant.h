#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::medical {

    /**
     * @brief Analysis of a tissue patch.
     */
    struct PathologyResult {
        // Color-coded segmentation map (Overlay ready)
        cv::Mat tissue_mask;

        // Breakdown of tissue types
        // e.g., "Tumor": 45.2%, "Stroma": 30.1%
        std::map<std::string, float> tissue_percentages;

        // Critical metric: Is this patch suspicious?
        bool has_malignancy;
        float tumor_burden; // 0.0 - 1.0
    };

    struct PathologyConfig {
        // Hardware Target (NVIDIA Clara / TRT is standard for medical)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., unet_gleason.engine)
        // Trained to output pixel-wise tissue class.
        std::string model_path;

        // Input Specs (Pathology models often use 256x256 or 512x512 patches)
        int input_width = 512;
        int input_height = 512;

        // Normalization
        // H&E images vary wildly. Standard approach is Macenko normalization
        // (often done externally) or standard mean/std scaling.
        std::vector<float> mean = {0.70f * 255, 0.70f * 255, 0.70f * 255}; // Approx H&E mean
        std::vector<float> std  = {0.20f * 255, 0.20f * 255, 0.20f * 255};

        // Tissue Classes
        // 0: Background/Glass, 1: Normal, 2: Tumor, 3: Stroma
        std::vector<std::string> class_names;
        std::vector<std::vector<uint8_t>> class_colors;

        // Alert Threshold
        float tumor_threshold = 0.05f; // Flag if > 5% tumor

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PathologyAssistant {
    public:
        explicit PathologyAssistant(const PathologyConfig& config);
        ~PathologyAssistant();

        // Move semantics
        PathologyAssistant(PathologyAssistant&&) noexcept;
        PathologyAssistant& operator=(PathologyAssistant&&) noexcept;
        PathologyAssistant(const PathologyAssistant&) = delete;
        PathologyAssistant& operator=(const PathologyAssistant&) = delete;

        /**
         * @brief Analyze a histology patch.
         *
         * Pipeline:
         * 1. Preprocess (Resize/Norm).
         * 2. Inference (Semantic Segmentation).
         * 3. Postprocess (ArgMax -> Tissue Map).
         * 4. Quantification (Calculate Tumor Burden).
         *
         * @param patch Input image (RGB).
         * @return Analysis metrics and visualization.
         */
        PathologyResult analyze_patch(const cv::Mat& patch);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical