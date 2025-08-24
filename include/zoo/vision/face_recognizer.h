#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    using FaceEmbedding = std::vector<float>;

    struct FaceRecognizerConfig {
        std::string engine_path;
        int input_width = 112;
        int input_height = 112;
    };

    class FaceRecognizer {
    public:
        explicit FaceRecognizer(const FaceRecognizerConfig& config);
        ~FaceRecognizer();

        FaceRecognizer(const FaceRecognizer&) = delete;
        FaceRecognizer& operator=(const FaceRecognizer&) = delete;
        FaceRecognizer(FaceRecognizer&&) noexcept;
        FaceRecognizer& operator=(FaceRecognizer&&) noexcept;

        FaceEmbedding predict(const cv::Mat& aligned_face_image);

        static float compare(const FaceEmbedding& emb1, const FaceEmbedding& emb2);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

