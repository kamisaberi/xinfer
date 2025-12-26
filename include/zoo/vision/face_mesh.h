#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief A single 3D landmark point.
     * Z is relative depth (usually normalized).
     */
    struct MeshPoint {
        float x;
        float y;
        float z;
    };

    struct FaceMeshResult {
        // The 468 dense landmarks
        std::vector<MeshPoint> landmarks;

        // Face presence confidence score
        float score;
    };

    struct FaceMeshConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., face_mesh.onnx)
        std::string model_path;

        // Input Specs (MediaPipe standard is 192x192 or 256x256)
        int input_width = 192;
        int input_height = 192;

        // Normalization (Standard [-1, 1] for MediaPipe models)
        std::vector<float> mean = {127.5f, 127.5f, 127.5f};
        std::vector<float> std  = {127.5f, 127.5f, 127.5f};
        float scale_factor = 1.0f; // Input 0-255

        // Threshold to consider a valid face
        float score_threshold = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class FaceMesh {
    public:
        explicit FaceMesh(const FaceMeshConfig& config);
        ~FaceMesh();

        // Move semantics
        FaceMesh(FaceMesh&&) noexcept;
        FaceMesh& operator=(FaceMesh&&) noexcept;
        FaceMesh(const FaceMesh&) = delete;
        FaceMesh& operator=(const FaceMesh&) = delete;

        /**
         * @brief Estimate face mesh from a face image.
         *
         * @param face_crop Input image (BGR). Should be a crop of the face.
         * @return Result containing 3D points.
         */
        FaceMeshResult estimate(const cv::Mat& face_crop);

        /**
         * @brief Utility: Draw the mesh points on the image.
         */
        static void draw_mesh(cv::Mat& image, const FaceMeshResult& result);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision