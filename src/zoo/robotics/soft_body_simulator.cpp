#include <xinfer/zoo/robotics/soft_body_simulator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Preproc skipped; custom Spatial Hashing implemented internally.

#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace xinfer::zoo::robotics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SoftBodySimulator::Impl {
    SimulatorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // System State
    std::vector<Particle> particles_;

    // Tensors (Graph Inputs)
    core::Tensor node_features; // [NumParticles, FeatDim] (Type, PosHistory)
    core::Tensor edge_index;    // [2, NumEdges] (Src, Dst)
    core::Tensor edge_attr;     // [NumEdges, AttrDim] (Relative Pos, Distance)

    // Tensor (Graph Output)
    core::Tensor output_accel;  // [NumParticles, 3] (Predicted Acceleration)

    Impl(const SimulatorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("SoftBodySimulator: Failed to load model " + config_.model_path);
        }

        // Note: Graph tensors change size every frame based on connectivity.
        // We defer resizing to the step() loop.
    }

    // --- Spatial Hashing for Neighbor Search ---
    // Finds all pairs of particles within 'connectivity_radius'
    void build_graph() {
        float r = config_.connectivity_radius;
        float cell_size = r;

        // Simple Hash: (x,y,z) -> int key
        auto hash_func = [&](int x, int y, int z) {
            return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
        };

        std::unordered_multimap<int, int> grid;
        grid.reserve(particles_.size());

        // 1. Insert particles into grid
        for (int i = 0; i < particles_.size(); ++i) {
            int cx = (int)(particles_[i].x / cell_size);
            int cy = (int)(particles_[i].y / cell_size);
            int cz = (int)(particles_[i].z / cell_size);
            grid.insert({hash_func(cx, cy, cz), i});
        }

        // 2. Query neighbors (27 cells around)
        std::vector<int> senders;
        std::vector<int> receivers;
        std::vector<float> attrs; // Relative displacement

        for (int i = 0; i < particles_.size(); ++i) {
            int cx = (int)(particles_[i].x / cell_size);
            int cy = (int)(particles_[i].y / cell_size);
            int cz = (int)(particles_[i].z / cell_size);

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        int key = hash_func(cx+dx, cy+dy, cz+dz);
                        auto range = grid.equal_range(key);

                        for (auto it = range.first; it != range.second; ++it) {
                            int j = it->second;
                            if (i == j) continue; // Self-loop handled by node feats usually

                            float rx = particles_[i].x - particles_[j].x;
                            float ry = particles_[i].y - particles_[j].y;
                            float rz = particles_[i].z - particles_[j].z;
                            float dist_sq = rx*rx + ry*ry + rz*rz;

                            if (dist_sq < r*r) {
                                senders.push_back(j);
                                receivers.push_back(i);

                                float dist = std::sqrt(dist_sq);
                                attrs.push_back(rx / dist); // Normalized Direction
                                attrs.push_back(ry / dist);
                                attrs.push_back(rz / dist);
                                attrs.push_back(dist);      // Magnitude
                            }
                        }
                    }
                }
            }
        }

        // 3. Fill Tensors
        size_t num_edges = senders.size();
        edge_index.resize({2, (int64_t)num_edges}, core::DataType::kINT64);
        edge_attr.resize({(int64_t)num_edges, 4}, core::DataType::kFLOAT);

        int64_t* idx_ptr = static_cast<int64_t*>(edge_index.data());
        float* attr_ptr = static_cast<float*>(edge_attr.data());

        // Copy data (could optimize this with direct writing in loop above)
        std::memcpy(idx_ptr, senders.data(), num_edges * sizeof(int64_t));
        std::memcpy(idx_ptr + num_edges, receivers.data(), num_edges * sizeof(int64_t));
        std::memcpy(attr_ptr, attrs.data(), num_edges * 4 * sizeof(float));

        // 4. Fill Node Features (Material ID + Velocity history)
        // Assuming model takes [Type, VelX, VelY, VelZ]
        int num_nodes = particles_.size();
        node_features.resize({(int64_t)num_nodes, 4}, core::DataType::kFLOAT);
        float* node_ptr = static_cast<float*>(node_features.data());

        for (int i = 0; i < num_nodes; ++i) {
            node_ptr[i * 4 + 0] = (float)particles_[i].material_id; // Embeddings usually inside network
            node_ptr[i * 4 + 1] = particles_[i].vx;
            node_ptr[i * 4 + 2] = particles_[i].vy;
            node_ptr[i * 4 + 3] = particles_[i].vz;
        }
    }

    // --- Semi-Implicit Euler Integration ---
    void integrate() {
        const float* accel = static_cast<const float*>(output_accel.data());
        float dt = config_.dt;

        for (size_t i = 0; i < particles_.size(); ++i) {
            // Read predicted acceleration
            float ax = accel[i * 3 + 0];
            float ay = accel[i * 3 + 1];
            float az = accel[i * 3 + 2];

            // Add Gravity (if not modeled by network)
            ay += config_.gravity_y;

            // Update Velocity
            particles_[i].vx += ax * dt;
            particles_[i].vy += ay * dt;
            particles_[i].vz += az * dt;

            // Update Position
            particles_[i].x += particles_[i].vx * dt;
            particles_[i].y += particles_[i].vy * dt;
            particles_[i].z += particles_[i].vz * dt;

            // Simple ground plane collision (y = 0)
            if (particles_[i].y < 0.0f) {
                particles_[i].y = 0.0f;
                particles_[i].vy = 0.0f; // Friction/Stop
            }
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

SoftBodySimulator::SoftBodySimulator(const SimulatorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SoftBodySimulator::~SoftBodySimulator() = default;
SoftBodySimulator::SoftBodySimulator(SoftBodySimulator&&) noexcept = default;
SoftBodySimulator& SoftBodySimulator::operator=(SoftBodySimulator&&) noexcept = default;

void SoftBodySimulator::set_state(const std::vector<Particle>& particles) {
    if (!pimpl_) throw std::runtime_error("SoftBodySimulator is null.");
    pimpl_->particles_ = particles;
}

std::vector<Particle> SoftBodySimulator::step(const BoundaryCondition& effector) {
    if (!pimpl_) throw std::runtime_error("SoftBodySimulator is null.");

    // 0. Update Boundary Particles
    // (In a real GNS, the gripper is represented as kinematic particles in the graph)
    // Here we might overwrite the last N particles to match the effector state.

    // 1. Build Graph (Neighbor Search)
    pimpl_->build_graph();

    // 2. Inference
    // Input: [Nodes, EdgeIndex, EdgeAttr]
    // Output: [Acceleration]
    pimpl_->engine_->predict(
        {pimpl_->node_features, pimpl_->edge_index, pimpl_->edge_attr},
        {pimpl_->output_accel}
    );

    // 3. Integrate
    pimpl_->integrate();

    return pimpl_->particles_;
}

} // namespace xinfer::zoo::robotics