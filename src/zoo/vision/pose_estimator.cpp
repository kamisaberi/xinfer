#include <include/zoo/vision/pose_estimator.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/zoo/vision/detector.h>

namespace xinfer::zoo::vision {

struct PoseEstimator::Impl {
    PoseEstimatorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

PoseEstimator::PoseEstimator(const PoseEstimatorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Pose estimation engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );
}

PoseEstimator::~PoseEstimator() = default;
PoseEstimator::PoseEstimator(PoseEstimator&&) noexcept = default;
PoseEstimator& PoseEstimator::operator=(PoseEstimator&&) noexcept = default;

std::vector<Pose> PoseEstimator::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("PoseEstimator is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& heatmap_tensor = output_tensors[0];

    auto heatmap_shape = heatmap_tensor.shape();
    const int num_kpts = heatmap_shape[1];
    const int H = heatmap_shape[2];
    const int W = heatmap_shape[3];

    std::vector<float> heatmaps(heatmap_tensor.num_elements());
    heatmap_tensor.copy_to_host(heatmaps.data());

    std::vector<Pose> poses;

    float scale_x = (float)image.cols / W;
    float scale_y = (float)image.rows / H;

    Pose current_pose;
    for (int k = 0; k < num_kpts; ++k) {
        float max_val = -1.0f;
        cv::Point max_loc(-1, -1);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float val = heatmaps[k * H * W + y * W + x];
                if (val > max_val) {
                    max_val = val;
                    max_loc = cv::Point(x, y);
                }
            }
        }

        Keypoint kpt;
        if (max_val > pimpl_->config_.keypoint_threshold) {
            kpt.x = (float)max_loc.x * scale_x;
            kpt.y = (float)max_loc.y * scale_y;
            kpt.confidence = max_val;
        } else {
            kpt.x = -1;
            kpt.y = -1;
            kpt.confidence = 0.0;
        }
        current_pose.push_back(kpt);
    }
    poses.push_back(current_pose);

    return poses;
}

} // namespace xinfer::zoo::vision