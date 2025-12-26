#include <xinfer/zoo/generative/outpainter.h>
#include <xinfer/core/logging.h>

// --- We reuse the Inpainter Zoo module ---
#include <xinfer/zoo/generative/inpainter.h>

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::generative {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Outpainter::Impl {
    OutpainterConfig config_;

    // The Inpainter module does all the heavy lifting.
    std::unique_ptr<Inpainter> inpainter_;

    Impl(const OutpainterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup the underlying Inpainter
        // We pass the same model paths to it.
        InpainterConfig inpaint_cfg;
        inpaint_cfg.target = config_.target;
        inpainter_ = std::make_unique<Inpainter>(inpaint_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

Outpainter::Outpainter(const OutpainterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Outpainter::~Outpainter() = default;
Outpainter::Outpainter(Outpainter&&) noexcept = default;
Outpainter& Outpainter::operator=(Outpainter&&) noexcept = default;

cv::Mat Outpainter::outpaint(const cv::Mat& image,
                             int top, int bottom, int left, int right,
                             const std::string& prompt) {
    if (!pimpl_ || !pimpl_->inpainter_) throw std::runtime_error("Outpainter is null.");

    // 1. Create a larger canvas
    int new_width = image.cols + left + right;
    int new_height = image.rows + top + bottom;

    cv::Mat canvas = cv::Mat::zeros(new_height, new_width, image.type());

    // 2. Place original image in the center
    cv::Rect roi(left, top, image.cols, image.rows);
    image.copyTo(canvas(roi));

    // 3. Create the Inpainting Mask
    // The mask is the INVERSE of the ROI. We want to fill everything *except* the center.
    cv::Mat mask = cv::Mat::ones(new_height, new_width, CV_8U) * 255;

    // Set the region where the original image is to 0 (don't touch)
    mask(roi).setTo(cv::Scalar(0));

    // 4. Call the Inpainter
    // The inpainter will fill the white areas of the mask.
    // The final image returned by the inpainter will have the expanded content.

    // For best results, outpainting is often done iteratively in smaller strips,
    // but a single-shot inpaint call is simpler to demonstrate.
    // To do that, the canvas+mask would need to be resized to the model's input size.
    // For this example, we assume the inpainter's internal preprocessor handles resizing.

    return pimpl_->inpainter_->inpaint(canvas, mask, prompt);
}

} // namespace xinfer::zoo::generative