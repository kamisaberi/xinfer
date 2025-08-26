#include <include/zoo/medical/pathology_assistant.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::medical {

struct PathologyAssistant::Impl {
    PathologyAssistantConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

PathologyAssistant::PathologyAssistant(const PathologyAssistantConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Pathology assistant engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.tile_size,
        pimpl_->config_.tile_size,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );
}

PathologyAssistant::~PathologyAssistant() = default;
PathologyAssistant::PathologyAssistant(PathologyAssistant&&) noexcept = default;
PathologyAssistant& PathologyAssistant::operator=(PathologyAssistant&&) noexcept = default;

void process_tile_batch(
    const std::vector<cv::Mat>& tile_batch,
    const std::vector<cv::Point>& tile_indices,
    cv::Mat& heatmap,
    PathologyAssistant::Impl* pimpl)
{
    if (tile_batch.empty()) return;

    auto input_shape = pimpl->engine_->get_input_shape(0);
    input_shape[0] = tile_batch.size();

    core::Tensor input_batch_tensor(input_shape, core::DataType::kFLOAT);

    std::vector<float> batch_data;
    batch_data.reserve(input_batch_tensor.num_elements());

    for(const auto& tile : tile_batch) {
        core::Tensor single_tile_tensor({1, input_shape[1], input_shape[2], input_shape[3]}, core::DataType::kFLOAT);
        pimpl->preprocessor_->process(tile, single_tile_tensor);

        std::vector<float> tile_data(single_tile_tensor.num_elements());
        single_tile_tensor.copy_to_host(tile_data.data());
        batch_data.insert(batch_data.end(), tile_data.begin(), tile_data.end());
    }
    input_batch_tensor.copy_from_host(batch_data.data());

    auto output_tensors = pimpl->engine_->infer({input_batch_tensor});
    const core::Tensor& scores_tensor = output_tensors[0];

    std::vector<float> scores(scores_tensor.num_elements());
    scores_tensor.copy_to_host(scores.data());

    for (size_t i = 0; i < tile_indices.size(); ++i) {
        heatmap.at<float>(tile_indices[i]) = scores[i];
    }
}

PathologyResult PathologyAssistant::predict(const cv::Mat& whole_slide_image) {
    if (!pimpl_) throw std::runtime_error("PathologyAssistant is in a moved-from state.");

    int tile_size = pimpl_->config_.tile_size;
    int batch_size = pimpl_->config_.batch_size;

    int num_tiles_x = whole_slide_image.cols / tile_size;
    int num_tiles_y = whole_slide_image.rows / tile_size;

    cv::Mat heatmap = cv::Mat::zeros(num_tiles_y, num_tiles_x, CV_32F);

    std::vector<cv::Mat> tile_batch;
    std::vector<cv::Point> tile_indices;
    tile_batch.reserve(batch_size);
    tile_indices.reserve(batch_size);

    for (int y = 0; y < num_tiles_y; ++y) {
        for (int x = 0; x < num_tiles_x; ++x) {
            cv::Rect roi(x * tile_size, y * tile_size, tile_size, tile_size);
            tile_batch.push_back(whole_slide_image(roi));
            tile_indices.push_back(cv::Point(x, y));

            if (tile_batch.size() >= batch_size) {
                process_tile_batch(tile_batch, tile_indices, heatmap, pimpl_.get());
                tile_batch.clear();
                tile_indices.clear();
            }
        }
    }

    process_tile_batch(tile_batch, tile_indices, heatmap, pimpl_.get());

    PathologyResult result;
    cv::Scalar avg_score = cv::mean(heatmap);
    result.overall_mitotic_score = avg_score[0];

    cv::Mat resized_heatmap;
    cv::resize(heatmap, resized_heatmap, whole_slide_image.size(), 0, 0, cv::INTER_LINEAR);
    result.mitotic_heatmap = resized_heatmap;

    return result;
}

} // namespace xinfer::zoo::medical