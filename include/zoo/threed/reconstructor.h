#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::threed {

    struct Mesh3D {
        std::vector<float> vertices;
        std::vector<int> faces;
        std::vector<unsigned char> vertex_colors;
    };

    struct ReconstructorConfig {
        int num_iterations = 10000;
        float quality = 1.0f;
    };

    class Reconstructor {
    public:
        explicit Reconstructor(const ReconstructorConfig& config);
        ~Reconstructor();

        Reconstructor(const Reconstructor&) = delete;
        Reconstructor& operator=(const Reconstructor&) = delete;
        Reconstructor(Reconstructor&&) noexcept;
        Reconstructor& operator=(Reconstructor&&) noexcept;

        Mesh3D predict(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& camera_poses);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed

