#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::accessibility {

    /**
     * @brief A high-level description of the scene.
     */
    struct SceneDescription {
        std::string narrative; // The final text summary
        bool has_obstacles;
        std::string immediate_obstacle_info; // e.g., "Stairs ahead"
        std::vector<std::string> readable_text; // All text found
    };

    struct AssistantConfig {
        // Hardware Target (Edge GPU like Jetson Orin is ideal for this multi-model pipeline)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // 1. General Object Detector (YOLO)
        std::string detector_path;
        std::string det_labels_path;

        // 2. Depth Estimator (MiDaS / DepthAnything)
        std::string depth_model_path;

        // 3. OCR Engine (Text Detector + Recognizer)
        std::string ocr_model_path;

        // 4. Captioning/VQA LLM (e.g., a small LLaVA or fine-tuned GPT-2)
        std::string llm_path;
        std::string llm_tokenizer_path;

        // --- Logic ---
        float obstacle_distance_m = 1.5f; // Alert if object is closer than this

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class VisualAssistant {
    public:
        explicit VisualAssistant(const AssistantConfig& config);
        ~VisualAssistant();

        // Move semantics
        VisualAssistant(VisualAssistant&&) noexcept;
        VisualAssistant& operator=(VisualAssistant&&) noexcept;
        VisualAssistant(const VisualAssistant&) = delete;
        VisualAssistant& operator=(const VisualAssistant&) = delete;

        /**
         * @brief Analyze a camera frame and generate a scene description.
         *
         * @param image Input camera frame.
         * @return A natural language summary and alerts.
         */
        SceneDescription describe_scene(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::accessibility