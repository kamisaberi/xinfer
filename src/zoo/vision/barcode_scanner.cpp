#include <include/zoo/vision/barcode_scanner.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

// For this specific task, we will use a CPU-based library for the final decoding
// as there are no standard GPU decoders. The AI model only finds the barcode location.
#include <zbar.h>

namespace xinfer::zoo::vision {

struct BarcodeScanner::Impl {
    BarcodeScannerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    zbar::ImageScanner zbar_scanner_;

    Impl(const BarcodeScannerConfig& config) : config_(config) {
        zbar_scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
    }
};

BarcodeScanner::BarcodeScanner(const BarcodeScannerConfig& config)
    : pimpl_(new Impl(config))
{
    if (!std::ifstream(pimpl_->config_.detection_engine_path).good()) {
        throw std::runtime_error("Barcode detection engine file not found: " + pimpl_->config_.detection_engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.detection_engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        true // Enable letterbox padding
    );
}

BarcodeScanner::~BarcodeScanner() = default;
BarcodeScanner::BarcodeScanner(BarcodeScanner&&) noexcept = default;
BarcodeScanner& BarcodeScanner::operator=(BarcodeScanner&&) noexcept = default;

std::vector<Barcode> BarcodeScanner::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("BarcodeScanner is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    // Assume the model is a detector that gives bounding boxes [x1, y1, x2, y2, conf, class]
    // A full implementation would use the NMS post-processor here.
    // For this example, we'll simplify and assume we get a list of boxes.
    std::vector<std::vector<float>> detected_boxes; // This would be populated from the model output

    std::vector<Barcode> results;

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    for (const auto& box_coords : detected_boxes) {
        if (box_coords[4] < pimpl_->config_.confidence_threshold) continue;

        cv::Rect roi(box_coords[0], box_coords[1], box_coords[2] - box_coords[0], box_coords[3] - box_coords[1]);
        roi &= cv::Rect(0, 0, gray.cols, gray.rows); // Clamp to image boundaries
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat patch = gray(roi);

        zbar::Image zbar_image(patch.cols, patch.rows, "Y800", patch.data, patch.total() * patch.elemSize());
        int n = pimpl_->zbar_scanner_.scan(zbar_image);

        if (n > 0) {
            for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin(); symbol != zbar_image.symbol_end(); ++symbol) {
                Barcode barcode;
                barcode.text = symbol->get_data();
                barcode.confidence = symbol->get_quality();

                switch(symbol->get_type()) {
                    case zbar::ZBAR_QRCODE: barcode.type = Barcode::Type::QR_CODE; break;
                    case zbar::ZBAR_CODE128: barcode.type = Barcode::Type::CODE_128; break;
                    case zbar::ZBAR_EAN13: barcode.type = Barcode::Type::EAN_13; break;
                    case zbar::ZBAR_UPCA: barcode.type = Barcode::Type::UPC_A; break;
                    case zbar::ZBAR_DATAMATRIX: barcode.type = Barcode::Type::DATA_MATRIX; break;
                    default: barcode.type = Barcode::Type::UNKNOWN; break;
                }

                for(int i = 0; i < symbol->get_location_size(); ++i) {
                    barcode.points.push_back({(float)symbol->get_location_x(i) + roi.x, (float)symbol->get_location_y(i) + roi.y});
                }
                results.push_back(barcode);
            }
        }
    }

    return results;
}

} // namespace xinfer::zoo::vision