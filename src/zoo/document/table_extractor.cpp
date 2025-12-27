#include <xinfer/zoo/document/table_extractor.h>
#include <xinfer/core/logging.h>

// --- We compose other Zoo modules ---
#include <xinfer/zoo/document/layout_parser.h>
#include <xinfer/zoo/vision/detector.h>
#include <xinfer/zoo/vision/ocr.h>

#include <iostream>
#include <algorithm>
#include <vector>

namespace xinfer::zoo::document {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TableExtractor::Impl {
    TableExtractorConfig config_;

    // --- Components ---
    std::unique_ptr<LayoutParser> layout_parser_;
    std::unique_ptr<vision::ObjectDetector> cell_detector_;
    std::unique_ptr<vision::OcrRecognizer> ocr_engine_;

    Impl(const TableExtractorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Init Layout Parser (to find the table region)
        LayoutParserConfig layout_cfg;
        layout_cfg.target = config_.target;
        layout_cfg.model_path = config_.layout_model_path;
        layout_cfg.labels_path = config_.layout_labels_path;
        layout_parser_ = std::make_unique<LayoutParser>(layout_cfg);

        // 2. Init Cell Detector
        vision::DetectorConfig cell_det_cfg;
        cell_det_cfg.target = config_.target;
        cell_det_cfg.model_path = config_.cell_model_path;
        cell_det_cfg.confidence_threshold = config_.cell_conf_thresh;
        // Cell detector is usually single-class ("Cell")
        cell_det_cfg.labels_path = "";
        cell_detector_ = std::make_unique<vision::ObjectDetector>(cell_det_cfg);

        // 3. Init OCR
        vision::OcrConfig ocr_cfg;
        ocr_cfg.target = config_.target;
        ocr_cfg.model_path = config_.ocr_model_path;
        ocr_cfg.vocab_path = config_.ocr_vocab_path;
        ocr_engine_ = std::make_unique<vision::OcrRecognizer>(ocr_cfg);
    }

    // --- Core Logic: Reconstruct Grid from Unordered Cells ---
    ExtractedTable reconstruct_table(const std::vector<TableCell>& cells) {
        ExtractedTable table;
        if (cells.empty()) return table;

        // 1. Sort cells by Y, then X coordinate (Top-to-bottom, Left-to-right)
        auto sorted_cells = cells;
        std::sort(sorted_cells.begin(), sorted_cells.end(), [](const TableCell& a, const TableCell& b) {
            // Allow a small tolerance in Y to group cells in the same row
            int y_tolerance = 10;
            if (std::abs(a.box.y - b.box.y) > y_tolerance) {
                return a.box.y < b.box.y;
            }
            return a.box.x < b.box.x;
        });

        // 2. Group into Rows
        std::vector<std::vector<TableCell>> rows;
        if (!sorted_cells.empty()) {
            rows.push_back({sorted_cells[0]});
            for (size_t i = 1; i < sorted_cells.size(); ++i) {
                // If current cell's Y is close to the last row's Y, add to it
                if (std::abs(sorted_cells[i].box.y - rows.back()[0].box.y) < 20) { // Row height tolerance
                    rows.back().push_back(sorted_cells[i]);
                } else {
                    rows.push_back({sorted_cells[i]});
                }
            }
        }

        // 3. Determine Column Boundaries
        // Simple method: Use median X coordinates of cells in the first row
        std::vector<float> col_boundaries;
        if (!rows.empty()) {
            for (const auto& cell : rows[0]) {
                col_boundaries.push_back((float)cell.box.x + cell.box.width / 2.0f);
            }
        }

        // 4. Build the 2D String Matrix
        table.data.resize(rows.size());
        for (size_t r = 0; r < rows.size(); ++r) {
            // Sort cells within this row by X to be sure
            std::sort(rows[r].begin(), rows[r].end(), [](const TableCell& a, const TableCell& b) {
                return a.box.x < b.box.x;
            });

            table.data[r].resize(col_boundaries.size(), ""); // Init with empty strings

            for (const auto& cell : rows[r]) {
                // Find which column this cell belongs to
                float cell_center_x = (float)cell.box.x + cell.box.width / 2.0f;
                int best_col = 0;
                float min_dist = 1e9;

                for (size_t c = 0; c < col_boundaries.size(); ++c) {
                    float dist = std::abs(cell_center_x - col_boundaries[c]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_col = c;
                    }
                }
                table.data[r][best_col] = cell.text;
            }
        }

        return table;
    }
};

// =================================================================================
// Public API
// =================================================================================

TableExtractor::TableExtractor(const TableExtractorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TableExtractor::~TableExtractor() = default;
TableExtractor::TableExtractor(TableExtractor&&) noexcept = default;
TableExtractor& TableExtractor::operator=(TableExtractor&&) noexcept = default;

std::vector<ExtractedTable> TableExtractor::extract(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("TableExtractor is null.");

    std::vector<ExtractedTable> all_tables;

    // 1. Find Table Regions
    auto layout = pimpl_->layout_parser_->parse(image);

    for (const auto& elem : layout.elements) {
        if (elem.type == "table") {
            cv::Rect table_roi(elem.box.x1, elem.box.y1, elem.box.x2 - elem.box.x1, elem.box.y2 - elem.box.y1);
            table_roi &= cv::Rect(0, 0, image.cols, image.rows);
            if (table_roi.width <= 0 || table_roi.height <= 0) continue;

            cv::Mat table_crop = image(table_roi);

            // 2. Detect Cells within the table crop
            auto cell_dets = pimpl_->cell_detector_->predict(table_crop);

            // 3. OCR each cell
            std::vector<TableCell> cells;
            for (const auto& cell_box : cell_dets) {
                TableCell cell;
                cell.box = cv::Rect(cell_box.x1, cell_box.y1, cell_box.x2-cell_box.x1, cell_box.y2-cell_box.y1);

                // OCR
                cv::Mat cell_crop = table_crop(cell.box);
                auto ocr_res = pimpl_->ocr_engine_->recognize(cell_crop);
                cell.text = ocr_res.text;
                cells.push_back(cell);
            }

            // 4. Reconstruct
            ExtractedTable table = pimpl_->reconstruct_table(cells);
            table.location_in_page = table_roi;
            all_tables.push_back(table);
        }
    }

    return all_tables;
}

} // namespace xinfer::zoo::document