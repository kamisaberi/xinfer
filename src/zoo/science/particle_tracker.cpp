#include <xinfer/zoo/science/particle_tracker.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Specialized physics math used here instead of generic preproc

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace xinfer::zoo::science {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ParticleTracker::Impl {
    ParticleTrackerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const ParticleTrackerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ParticleTracker: Failed to load model " + config_.model_path);
        }

        // 2. Pre-allocate Input Tensor
        // Shape: [1, MaxHits, Features]
        input_tensor.resize({1, (int64_t)config_.max_hits, (int64_t)config_.input_features}, core::DataType::kFLOAT);
    }

    // --- Math: Coordinate Transform & Normalization ---
    void process_hits(const std::vector<DetectorHit>& hits) {
        float* ptr = static_cast<float*>(input_tensor.data());

        // Zero out buffer (Padding)
        std::memset(ptr, 0, input_tensor.size() * sizeof(float));

        int count = std::min((int)hits.size(), config_.max_hits);

        float inv_r_max = 1.0f / config_.detector_radius;
        float inv_z_max = 1.0f / config_.detector_length;

        for (int i = 0; i < count; ++i) {
            const auto& h = hits[i];

            // Cartesian -> Cylindrical
            float r = std::sqrt(h.x*h.x + h.y*h.y);
            float phi = std::atan2(h.y, h.x); // [-PI, PI]
            float z = h.z;

            // Normalize to [-1, 1] (or [0, 1]) for NN stability
            // Features: [r, phi, z]
            float norm_r = r * inv_r_max;
            float norm_phi = phi / M_PI; // Normalize to [-1, 1]
            float norm_z = z * inv_z_max;

            // Fill Tensor (Row Major)
            int offset = i * config_.input_features;
            ptr[offset + 0] = norm_r;
            ptr[offset + 1] = norm_phi;
            ptr[offset + 2] = norm_z;
        }
    }

    // --- Output Decoding ---
    std::vector<ParticleTrack> decode_output() {
        std::vector<ParticleTrack> tracks;

        // Assuming Model Output: [1, MaxTracks, 6]
        // Channels: [Pt, Eta, Phi, Charge, VertexZ, Score]
        const float* out_data = static_cast<const float*>(output_tensor.data());
        auto shape = output_tensor.shape();

        int num_possible_tracks = (int)shape[1];
        int num_params = (int)shape[2]; // Should be 6

        for (int i = 0; i < num_possible_tracks; ++i) {
            const float* params = out_data + (i * num_params);

            float score = params[5];

            if (score > config_.track_score_thresh) {
                ParticleTrack t;
                t.id = i;

                // Denormalization might be needed here depending on training
                // Assuming model predicts raw physical values or normalized 0-1
                t.pt = params[0];        // GeV
                t.eta = params[1];
                t.phi = params[2] * M_PI; // Restore scale if normalized
                t.charge = (params[3] > 0.5f) ? 1.0f : -1.0f; // Binary classification
                t.vertex_z = params[4] * config_.detector_length;
                t.confidence = score;

                tracks.push_back(t);
            }
        }
        return tracks;
    }
};

// =================================================================================
// Public API
// =================================================================================

ParticleTracker::ParticleTracker(const ParticleTrackerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ParticleTracker::~ParticleTracker() = default;
ParticleTracker::ParticleTracker(ParticleTracker&&) noexcept = default;
ParticleTracker& ParticleTracker::operator=(ParticleTracker&&) noexcept = default;

std::vector<ParticleTrack> ParticleTracker::reconstruct_event(const std::vector<DetectorHit>& event_hits) {
    if (!pimpl_) throw std::runtime_error("ParticleTracker is null.");

    // 1. Preprocess (XYZ -> Normalized Cylindrical Tensor)
    pimpl_->process_hits(event_hits);

    // 2. Inference
    // If running on FPGA, this pushes data via DMA/PCIe
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Tensor -> Tracks)
    return pimpl_->decode_output();
}

} // namespace xinfer::zoo::science