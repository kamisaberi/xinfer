#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::document {

    /**
     * @brief A single detected layout element.
     */
    struct LayoutElement {
        std::string type; // "paragraph", "title", "table", "figure", "list"
        float confidence;
        postproc::BoundingBox box;
    };

    struct LayoutResult {
        std::vector<LayoutElement> elements;

        // Visualization
        cv::Mat annotated_image;
    };

    struct LayoutParserConfig {
        // Hardware Target (CPU/iGPU is often sufficient for document models)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8_layout.onnx)
        std::string model_path;

        // Label Map (Class ID -> Element Type)
        std::string labels_path;

        // Input Specs (High resolution is needed for document structure)
        int input_width = 1024;
        int input_height = 1024;

        // Detection Settings
        float conf_threshold = 0.5f;
        float nms_threshold = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class LayoutParser {
    public:
        explicit LayoutParser(const LayoutParserConfig& config);
        ~LayoutParser();

        // Move semantics
        LayoutParser(LayoutParser&&) noexcept;
        LayoutParser& operator=(LayoutParser&&) noexcept;
        LayoutParser(const LayoutParser&) = delete;
        LayoutParser& operator=(const LayoutParser&) = delete;

        /**
         * @brief Parse the layout of a document page.
         *
         * @param image Input scan/image of the document.
         * @return A list of all detected structural elements.
         */
        LayoutResult parse(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document