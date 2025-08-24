#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    using ImageEmbedding = std::vector<float>;

    struct ImageSimilarityConfig {
        std::string engine_path;
        int input_width = 224;
        int input_height = 224;
    };

    class ImageSimilarity {
    public:
        explicit ImageSimilarity(const ImageSimilarityConfig& config);
        ~ImageSimilarity();

        ImageSimilarity(const ImageSimilarity&) = delete;
        ImageSimilarity& operator=(const ImageSimilarity&) = delete;
        ImageSimilarity(ImageSimilarity&&) noexcept;
        ImageSimilarity& operator=(ImageSimilarity&&) noexcept;

        ImageEmbedding predict(const cv::Mat& image);

        static float compare(const ImageEmbedding& emb1, const ImageEmbedding& emb2);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

