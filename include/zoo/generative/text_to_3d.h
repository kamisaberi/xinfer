#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct Vertex {
        float x, y, z;
    };

    struct Face {
        int v1, v2, v3;
    };

    struct Mesh3D {
        std::vector<Vertex> vertices;
        std::vector<Face> faces;

        /**
         * @brief Export to OBJ format for viewing in Blender/MeshLab.
         */
        bool save_obj(const std::string& filename) const;
    };

    struct TextTo3DConfig {
        // Hardware Target (High-end GPU with >16GB VRAM required)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // (Often a multi-model pipeline)
        std::string text_encoder_path;
        std::string shape_generator_path; // The main Transformer/Diffusion model
        // std::string sdf_decoder_path;  // Optional, if using implicit representation

        // --- Tokenizer ---
        std::string vocab_path;
        std::string merges_path;

        // --- Generation Parameters ---
        int num_inference_steps = 64; // For Diffusion-based generators
        float guidance_scale = 15.0f;
        int grid_resolution = 128; // Resolution for Marching Cubes
        int seed = -1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    /**
     * @brief Progress callback for long generation tasks.
     * @param step Current step.
     * @param total_steps Total steps.
     * @param status_message e.g., "Generating SDF Grid", "Running Marching Cubes".
     */
    using Progress3DCallback = std::function<void(int step, int total_steps, const std::string& status_message)>;

    class TextTo3D {
    public:
        explicit TextTo3D(const TextTo3DConfig& config);
        ~TextTo3D();

        // Move semantics
        TextTo3D(TextTo3D&&) noexcept;
        TextTo3D& operator=(TextTo3D&&) noexcept;
        TextTo3D(const TextTo3D&) = delete;
        TextTo3D& operator=(const TextTo3D&) = delete;

        /**
         * @brief Generate a 3D mesh from a text prompt.
         *
         * @param prompt The text description (e.g., "A high-quality 3d model of a hamburger").
         * @param callback Optional progress callback.
         * @return The final 3D mesh.
         */
        Mesh3D generate(const std::string& prompt, Progress3DCallback callback = nullptr);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative