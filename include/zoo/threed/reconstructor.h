#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::threed {

    struct Vertex {
        float x, y, z;
        uint8_t r, g, b;
    };

    struct Mesh {
        std::vector<Vertex> vertices;
        // Optional: std::vector<int> faces; // For surface reconstruction

        /**
         * @brief Export to PLY format (readable by MeshLab/Blender).
         */
        void save_ply(const std::string& filename) const;
    };

    struct ReconstructorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Depth Model Path (e.g., depth_anything.engine)
        std::string model_path;

        // Input Specs
        int input_width = 518; // Depth Anything standard
        int input_height = 518;

        // Camera Intrinsics (Pinhole Model)
        // Required to map 2D pixels to 3D space accurately.
        // Defaults are generic approximations.
        float fx = 1000.0f; // Focal Length X
        float fy = 1000.0f; // Focal Length Y
        float cx = 320.0f;  // Principal Point X (usually width/2)
        float cy = 240.0f;  // Principal Point Y (usually height/2)

        // Depth Scaling
        float depth_scale = 10.0f; // Multiplier to convert model output to meters
        float min_depth = 0.1f;    // Clip min
        float max_depth = 100.0f;  // Clip max

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Reconstructor {
    public:
        explicit Reconstructor(const ReconstructorConfig& config);
        ~Reconstructor();

        // Move semantics
        Reconstructor(Reconstructor&&) noexcept;
        Reconstructor& operator=(Reconstructor&&) noexcept;
        Reconstructor(const Reconstructor&) = delete;
        Reconstructor& operator=(const Reconstructor&) = delete;

        /**
         * @brief Reconstruct 3D scene from a single image.
         *
         * Pipeline:
         * 1. Estimate Depth Map (AI).
         * 2. Back-project pixels to 3D rays.
         * 3. Texture map using RGB colors.
         *
         * @param image Input RGB image.
         * @return 3D Mesh/Pointcloud.
         */
        Mesh reconstruct(const cv::Mat& image);

        /**
         * @brief Update camera intrinsics at runtime (e.g. zoom lens).
         */
        void set_intrinsics(float fx, float fy, float cx, float cy);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed