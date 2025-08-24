#include <include/zoo/generative/text_to_3d.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
// This task would require a text tokenizer, which would be another core component
// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::generative {

struct TextTo3D::Impl {
    TextTo3DConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    // std::unique_ptr<preproc::Tokenizer> tokenizer_;
};

TextTo3D::TextTo3D(const TextTo3DConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("TextTo3D engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::Tokenizer>("path/to/vocab.json");
}

TextTo3D::~TextTo3D() = default;
TextTo3D::TextTo3D(TextTo3D&&) noexcept = default;
TextTo3D& operator=(TextTo3D&&) noexcept = default;

Mesh3D TextTo3D::predict(const std::string& text_prompt) {
    if (!pimpl_) throw std::runtime_error("TextTo3D is in a moved-from state.");

    // Tokenize text prompt
    // std::vector<int> tokenized_text = pimpl_->tokenizer_->encode(text_prompt);
    // core::Tensor input_tensor({1, (int64_t)tokenized_text.size()}, core::DataType::kINT32);
    // input_tensor.copy_from_host(tokenized_text.data());

    // auto output_tensors = pimpl_->engine_->infer({input_tensor});

    // The output of a TextTo3D model could be in many formats.
    // Here we assume it outputs three tensors: vertices, faces, and colors.
    // const core::Tensor& vertices_tensor = output_tensors[0];
    // const core::Tensor& faces_tensor = output_tensors[1];
    // const core::Tensor& colors_tensor = output_tensors[2];

    Mesh3D result_mesh;
    // vertices_tensor.copy_to_host(result_mesh.vertices.data());
    // faces_tensor.copy_to_host(result_mesh.faces.data());
    // colors_tensor.copy_to_host(result_mesh.vertex_colors.data());

    return result_mesh;
}

} // namespace xinfer::zoo::generative