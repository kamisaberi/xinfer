#include <xinfer/zoo/generative/dcgan.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// No heavy preproc/postproc needed, just math

#include <iostream>
#include <random>

namespace xinfer::zoo::generative {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DCGAN::Impl {
    DcganConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Random Number Generator
    std::mt19937 rng_;
    std::normal_distribution<float> dist_;

    Impl(const DcganConfig& config) : config_(config), dist_(0.0f, 1.0f) {
        if (config.seed != -1) {
            rng_.seed(config.seed);
        } else {
            std::random_device rd;
            rng_.seed(rd());
        }

        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DCGAN: Failed to load model " + config_.model_path);
        }
    }

    // --- Preprocessing: Generate Random Noise ---
    void prepare_input(const std::vector<float>* custom_vector = nullptr) {
        // Input Shape: [1, LatentDim]
        input_tensor.resize({1, (int64_t)config_.latent_dim}, core::DataType::kFLOAT);
        float* ptr = static_cast<float*>(input_tensor.data());

        if (custom_vector) {
            std::memcpy(ptr, custom_vector->data(), config_.latent_dim * sizeof(float));
        } else {
            for (int i = 0; i < config_.latent_dim; ++i) {
                ptr[i] = dist_(rng_);
            }
        }
    }

    // --- Post-processing: Tensor -> Image ---
    cv::Mat tensor_to_image() {
        // Output Shape: [1, Channels, H, W]
        auto shape = output_tensor.shape();
        int c = (int)shape[1];
        int h = (int)shape[2];
        int w = (int)shape[3];
        int spatial = h * w;

        const float* data = static_cast<const float*>(output_tensor.data());

        // Denormalize: Tanh output is [-1, 1], convert to [0, 255]
        // val_uint8 = (val_float * 0.5 + 0.5) * 255
        std::vector<cv::Mat> channels;
        for (int i = 0; i < c; ++i) {
            // Wrap channel plane (no copy)
            cv::Mat float_chan(h, w, CV_32F, const_cast<float*>(data + i * spatial));

            // Denormalize and convert to uint8
            cv::Mat uint_chan;
            float_chan.convertTo(uint_chan, CV_8U, 255.0 * 0.5, 255.0 * 0.5);
            channels.push_back(uint_chan);
        }

        // Merge to BGR
        cv::Mat final_image;
        if (c == 3) {
            // Swap RGB -> BGR
            std::swap(channels[0], channels[2]);
            cv::merge(channels, final_image);
        } else if (c == 1) {
            final_image = channels[0];
        } else {
            // Fallback for other channel counts
            XINFER_LOG_WARN("DCGAN output has unexpected channel count: " + std::to_string(c));
            return cv::Mat();
        }

        return final_image;
    }
};

// =================================================================================
// Public API
// =================================================================================

DCGAN::DCGAN(const DcganConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DCGAN::~DCGAN() = default;
DCGAN::DCGAN(DCGAN&&) noexcept = default;
DCGAN& DCGAN::operator=(DCGAN&&) noexcept = default;

cv::Mat DCGAN::generate() {
    if (!pimpl_) throw std::runtime_error("DCGAN is null.");

    // 1. Prepare Input (Random Vector)
    pimpl_->prepare_input();

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    return pimpl_->tensor_to_image();
}

cv::Mat DCGAN::generate_from_vector(const std::vector<float>& latent_vector) {
    if (!pimpl_) throw std::runtime_error("DCGAN is null.");
    if (latent_vector.size() != (size_t)pimpl_->config_.latent_dim) {
        XINFER_LOG_ERROR("Latent vector size mismatch.");
        return cv::Mat();
    }

    // 1. Prepare Input (User-provided Vector)
    pimpl_->prepare_input(&latent_vector);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    return pimpl_->tensor_to_image();
}

} // namespace xinfer::zoo::generative