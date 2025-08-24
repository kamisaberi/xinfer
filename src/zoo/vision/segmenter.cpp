#include <include/zoo/vision/segmenter.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/segmentation.h>

namespace xinfer::zoo::vision {

    struct Segmenter::Impl {
        SegmenterConfig config_;
        std::unique_ptr<core::InferenceEngine> engine_;
        std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    };

    Segmenter::Segmenter(const SegmenterConfig& config)
        : pimpl_(new Impl{config})
    {
        if (!std::ifstream(pimpl_->config_.engine_path).good()) {
            throw std::runtime_error("Segmentation engine file not found: " + pimpl_->config_.engine_path);
        }

        pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

        pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
            pimpl_->config_.input_width,
            pimpl_->config_.input_height,
            std::vector<float>{0.485f, 0.456f, 0.406f},
            std::vector<float>{0.229f, 0.224f, 0.225f}
        );
    }

    Segmenter::~Segmenter() = default;
    Segmenter::Segmenter(Segmenter&&) noexcept = default;
    Segmenter& Segmenter::operator=(Segmenter&&) noexcept = default;

    cv::Mat Segmenter::predict(const cv::Mat& image) {
        if (!pimpl_) throw std::runtime_error("Segmenter is in a moved-from state.");

        auto input_shape = pimpl_->engine_->get_input_shape(0);
        core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
        pimpl_->preprocessor_->process(image, input_tensor);

        auto output_tensors = pimpl_->engine_->infer({input_tensor});
        const core::Tensor& logit_map_tensor = output_tensors[0];

        cv::Mat class_mask = postproc::argmax_to_mat(logit_map_tensor);

        cv::Mat final_mask;
        cv::resize(class_mask, final_mask, image.size(), 0, 0, cv::INTER_NEAREST);

        return final_mask;
    }

} // namespace xinfer::zoo::vision