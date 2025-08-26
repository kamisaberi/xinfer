#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::medical {

    struct PathologyResult {
        float overall_mitotic_score;
        cv::Mat mitotic_heatmap;
    };

    struct PathologyAssistantConfig {
        std::string engine_path;
        int tile_size = 256;
        int batch_size = 16;
        float score_threshold = 0.5f;
    };

    class PathologyAssistant {
    public:
        explicit PathologyAssistant(const PathologyAssistantConfig& config);
        ~PathologyAssistant();

        PathologyAssistant(const PathologyAssistant&) = delete;
        PathologyAssistant& operator=(const PathologyAssistant&) = delete;
        PathologyAssistant(PathologyAssistant&&) noexcept;
        PathologyAssistant& operator=(PathologyAssistant&&) noexcept;

        PathologyResult predict(const cv::Mat& whole_slide_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical


