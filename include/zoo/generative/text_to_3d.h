#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::core { class Tensor; }

namespace xinfer::zoo::generative {

    struct Mesh3D {
        std::vector<float> vertices; // [x1, y1, z1, x2, y2, z2, ...]
        std::vector<int> faces;      // [v1_idx, v2_idx, v3_idx, ...]
        std::vector<float> vertex_colors; // [r1, g1, b1, r2, g2, b2, ...]
    };

    struct TextTo3DConfig {
        std::string engine_path;
        // Add any specific config parameters for the 3D model here
    };

    class TextTo3D {
    public:
        explicit TextTo3D(const TextTo3DConfig& config);
        ~TextTo3D();

        TextTo3D(const TextTo3D&) = delete;
        TextTo3D& operator=(const TextTo3D&) = delete;
        TextTo3D(TextTo3D&&) noexcept;
        TextTo3D& operator=(TextTo3D&&) noexcept;

        Mesh3D predict(const std::string& text_prompt);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

