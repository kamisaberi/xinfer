#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <include/zoo/vision/ocr.h>

namespace xinfer::zoo::document {

    using Table = std::vector<std::vector<std::string>>;

    struct TableExtractorConfig {
        std::string structure_engine_path;
        std::string ocr_config_path; // Path to a config file for the OCR model
    };

    class TableExtractor {
    public:
        explicit TableExtractor(const TableExtractorConfig& config);
        ~TableExtractor();

        TableExtractor(const TableExtractor&) = delete;
        TableExtractor& operator=(const TableExtractor&) = delete;
        TableExtractor(TableExtractor&&) noexcept;
        TableExtractor& operator=(TableExtractor&&) noexcept;

        Table predict(const cv::Mat& table_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document

