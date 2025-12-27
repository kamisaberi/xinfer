#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::document {

    /**
     * @brief A single cell in the table.
     */
    struct TableCell {
        std::string text;
        int row;
        int col;
        cv::Rect box;
    };

    /**
     * @brief The full extracted table.
     * Represented as a 2D vector of strings (Rows x Columns).
     */
    struct ExtractedTable {
        std::vector<std::vector<std::string>> data;
        float confidence;
        cv::Rect location_in_page;
    };

    struct TableExtractorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Table Detector (Layout Parser) ---
        std::string layout_model_path;
        std::string layout_labels_path;

        // --- Model 2: Cell Detector ---
        // (YOLO trained on table cells)
        std::string cell_model_path;

        // --- Model 3: OCR ---
        std::string ocr_model_path;
        std::string ocr_vocab_path;

        // Input Specs
        int input_width = 1024;
        int input_height = 1024;

        // Logic
        float cell_conf_thresh = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class TableExtractor {
    public:
        explicit TableExtractor(const TableExtractorConfig& config);
        ~TableExtractor();

        // Move semantics
        TableExtractor(TableExtractor&&) noexcept;
        TableExtractor& operator=(TableExtractor&&) noexcept;
        TableExtractor(const TableExtractor&) = delete;
        TableExtractor& operator=(const TableExtractor&) = delete;

        /**
         * @brief Extract all tables from a document page.
         *
         * @param image Input image of the document.
         * @return A list of structured tables.
         */
        std::vector<ExtractedTable> extract(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document