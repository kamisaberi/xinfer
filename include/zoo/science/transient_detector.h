#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::science {

    /**
     * @brief A detected transient event.
     */
    struct TransientEvent {
        float x, y;          // Centroid position
        float flux;          // Integrated intensity (brightness change)
        float confidence;    // ML confidence "Real" vs "Bogus"
        postproc::BoundingBox box;
    };

    struct TransientConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., real_bogus_net.onnx)
        // A classifier that takes a crop and says "Real" or "Noise"
        std::string model_path;

        // Input Specs for the Classifier
        int crop_size = 64; // Size of the cutout around a candidate

        // Detection Thresholds (Classical)
        float diff_threshold = 30.0f; // Pixel intensity change > 30 (approx 3-5 sigma)
        int min_area = 5;             // Minimum pixels for a candidate

        // ML Threshold
        float ml_confidence_thresh = 0.8f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class TransientDetector {
    public:
        explicit TransientDetector(const TransientConfig& config);
        ~TransientDetector();

        // Move semantics
        TransientDetector(TransientDetector&&) noexcept;
        TransientDetector& operator=(TransientDetector&&) noexcept;
        TransientDetector(const TransientDetector&) = delete;
        TransientDetector& operator=(const TransientDetector&) = delete;

        /**
         * @brief Set the reference (template) image.
         * This image represents the "static" sky/scene.
         */
        void set_reference(const cv::Mat& ref_image);

        /**
         * @brief Process a new science image.
         *
         * Pipeline:
         * 1. Compute Difference (Science - Reference).
         * 2. Find Candidates (blobs in difference image).
         * 3. Validate Candidates using ML model (Real-Bogus classification).
         *
         * @param image Current observation (Must match reference size).
         * @return List of confirmed transient events.
         */
        std::vector<TransientEvent> detect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::science