#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::science {

    /**
     * @brief A raw hit on a detector layer.
     */
    struct DetectorHit {
        float x, y, z;       // Cartesian coordinates (mm)
        float energy_loss;   // dE/dx (optional)
        int layer_id;        // Which detector layer recorded this
    };

    /**
     * @brief A reconstructed particle track.
     */
    struct ParticleTrack {
        int id;
        float pt;            // Transverse Momentum (GeV)
        float eta;           // Pseudorapidity (Angle)
        float phi;           // Azimuthal Angle
        float charge;        // +1 or -1
        float vertex_z;      // Origin Z coordinate
        float confidence;    // Is this a real track?
    };

    struct ParticleTrackerConfig {
        // Hardware Target
        // FPGAs (AMD/Intel) are standard for L1 Triggers due to low latency.
        xinfer::Target target = xinfer::Target::AMD_VITIS;

        // Model Path (e.g., graph_track_net.xmodel)
        std::string model_path;

        // Input Constraints
        // Models usually accept a fixed max number of hits per sector/event.
        int max_hits = 1000;
        int input_features = 3; // (r, phi, z) is standard for HEP models

        // Physics Constants (for Normalization)
        float detector_radius = 1000.0f; // Max radius of tracker (mm)
        float detector_length = 3000.0f; // Max length z (mm)
        float magnetic_field = 3.8f;     // Tesla (used for Pt calculation validation)

        // Thresholds
        float track_score_thresh = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ParticleTracker {
    public:
        explicit ParticleTracker(const ParticleTrackerConfig& config);
        ~ParticleTracker();

        // Move semantics
        ParticleTracker(ParticleTracker&&) noexcept;
        ParticleTracker& operator=(ParticleTracker&&) noexcept;
        ParticleTracker(const ParticleTracker&) = delete;
        ParticleTracker& operator=(const ParticleTracker&) = delete;

        /**
         * @brief Reconstruct tracks from a set of hits.
         *
         * Pipeline:
         * 1. Preprocess: Coordinate Transform (XYZ -> Cylindrical).
         * 2. Normalization: Scale to [-1, 1].
         * 3. Inference: GNN/Transformer.
         * 4. Postprocess: Denormalize track parameters.
         *
         * @param event_hits Raw hits from the detector event.
         * @return List of reconstructed tracks.
         */
        std::vector<ParticleTrack> reconstruct_event(const std::vector<DetectorHit>& event_hits);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::science