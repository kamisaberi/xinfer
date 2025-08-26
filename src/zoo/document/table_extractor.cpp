#include <include/zoo/document/table_extractor.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::document {

    struct TableExtractor::Impl {
        TableExtractorConfig config_;
        std::unique_ptr<core::InferenceEngine> structure_engine_;
        std::unique_ptr<OCR> ocr_pipeline_;
    };

    TableExtractor::TableExtractor(const TableExtractorConfig& config)
        : pimpl_(new Impl{config})
    {
        if (!std::ifstream(pimpl_->config_.structure_engine_path).good()) {
            throw std::runtime_error("Table structure engine file not found: " + pimpl_->config_.structure_engine_path);
        }

        pimpl_->structure_engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.structure_engine_path);

        // OCRConfig ocr_config;
        // Load ocr_config from pimpl_->config_.ocr_config_path
        // pimpl_->ocr_pipeline_ = std::make_unique<OCR>(ocr_config);
    }

    TableExtractor::~TableExtractor() = default;
    TableExtractor::TableExtractor(TableExtractor&&) noexcept = default;
    TableExtractor& TableExtractor::operator=(TableExtractor&&) noexcept = default;

    Table TableExtractor::predict(const cv::Mat& table_image) {
        if (!pimpl_) throw std::runtime_error("TableExtractor is in a moved-from state.");

        // A full implementation would be a complex multi-stage pipeline:
        // 1. Run a detection model (the structure_engine_) to find row/column bounding boxes.
        // 2. Run the OCR pipeline on the full image to get all text and its location.
        // 3. A complex CPU-based algorithm to match the text results to the row/column boxes
        //    to reconstruct the final table structure.

        Table reconstructed_table;

        return reconstructed_table;
    }

} // namespace xinfer::zoo::document