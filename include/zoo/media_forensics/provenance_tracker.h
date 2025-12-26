#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::media_forensics {

    /**
     * @brief Result of a provenance check.
     */
    struct ProvenanceResult {
        // --- Content Retrieval (Fingerprinting) ---
        bool is_known_content;      // True if similarity > threshold
        std::string origin_id;      // ID of the original asset (from database)
        float similarity_score;     // 0.0 to 1.0 (Cosine Similarity)

        // --- Watermark decoding (Optional) ---
        bool has_watermark;
        std::string watermark_payload; // Decoded string/bits
    };

    struct ProvenanceConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Perceptual Hasher / Embedder ---
        // (e.g., simclr_resnet50.onnx or dinov2_small.engine)
        std::string embedder_path;
        int input_width = 224;
        int input_height = 224;

        // --- Model 2: Watermark Decoder (Optional) ---
        // (e.g., hidden_decoder.onnx)
        std::string decoder_path;

        // Database
        // Threshold to consider content a "Match" (Near-Duplicate)
        float match_threshold = 0.85f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ProvenanceTracker {
    public:
        explicit ProvenanceTracker(const ProvenanceConfig& config);
        ~ProvenanceTracker();

        // Move semantics
        ProvenanceTracker(ProvenanceTracker&&) noexcept;
        ProvenanceTracker& operator=(ProvenanceTracker&&) noexcept;
        ProvenanceTracker(const ProvenanceTracker&) = delete;
        ProvenanceTracker& operator=(const ProvenanceTracker&) = delete;

        /**
         * @brief Register an original asset into the local database.
         * Calculates and stores the fingerprint.
         *
         * @param id Unique identifier (filename, URL, hash).
         * @param image The original image.
         */
        void register_asset(const std::string& id, const cv::Mat& image);

        /**
         * @brief Analyze an image to find its source.
         *
         * Pipeline:
         * 1. Extract Fingerprint (Embedding).
         * 2. Search Database (Cosine Similarity).
         * 3. (Optional) Run Watermark Decoder.
         *
         * @param image Query image (potentially cropped/resized/compressed).
         * @return Provenance info.
         */
        ProvenanceResult trace(const cv::Mat& image);

        /**
         * @brief Clear the known assets database.
         */
        void clear_database();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::media_forensics