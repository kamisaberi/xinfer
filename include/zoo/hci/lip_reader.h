#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::hci {

    struct LipReadResult {
        std::string transcribed_text;
        float confidence; // Word-level confidence is complex, this is a sequence-level proxy
    };

    struct LipReaderConfig {
        // Hardware Target (3D-CNNs are heavy, GPU recommended)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: Face/Mouth Detector ---
        // Used to find the mouth ROI.
        std::string detector_path;

        // --- Model 2: Lip Reading Model ---
        // e.g., lipnet_gru.engine
        std::string lip_read_model_path;

        // --- Specs for Lip Reading Model ---
        int input_width = 120; // Width of mouth crop
        int input_height = 60;  // Height of mouth crop
        int window_size = 29;   // Temporal depth (frames)

        // --- Decoder ---
        std::string vocab_path; // Character vocabulary for CTC
        int blank_index = 0;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class LipReader {
    public:
        explicit LipReader(const LipReaderConfig& config);
        ~LipReader();

        // Move semantics
        LipReader(LipReader&&) noexcept;
        LipReader& operator=(LipReader&&) noexcept;
        LipReader(const LipReader&) = delete;
        LipReader& operator=(const LipReader&) = delete;

        /**
         * @brief Process a video frame.
         *
         * This function updates an internal buffer. Inference is only triggered
         * once enough frames are collected.
         *
         * @param image Input video frame.
         * @return Decoded text (only valid when a word/phrase is complete).
         */
        LipReadResult process_frame(const cv::Mat& image);

        /**
         * @brief Reset internal buffers.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::hci