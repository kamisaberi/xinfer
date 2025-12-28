#include <xinfer/zoo/chemistry/reaction_forecaster.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <numeric>

namespace xinfer::zoo::chemistry {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ReactionForecaster::Impl {
    ForecasterConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor node_features;
    core::Tensor adj_matrix; // Adjacency matrix for input

    // Outputs from the model
    core::Tensor out_node_features;
    core::Tensor out_adj_matrix;
    core::Tensor out_yield;

    Impl(const ForecasterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ReactionForecaster: Failed to load model.");
        }
    }

    // --- Preprocessing: Merge multiple reactants into one graph ---
    void prepare_input(const std::vector<MoleculeGraph>& reactants) {
        // 1. Count total atoms and prepare offsets
        int total_atoms = 0;
        std::vector<int> atom_offsets;
        for (const auto& mol : reactants) {
            atom_offsets.push_back(total_atoms);
            total_atoms += mol.atoms.size();
        }

        if (total_atoms > config_.max_atoms) {
            XINFER_LOG_WARN("Total atoms exceed max_atoms. Truncating.");
            total_atoms = config_.max_atoms;
        }

        // 2. Node Features Tensor [1, MaxAtoms, NumFeatures]
        node_features.resize({1, (int64_t)config_.max_atoms, (int64_t)config_.atom_feature_dim}, core::DataType::kFLOAT);
        float* node_ptr = static_cast<float*>(node_features.data());
        std::memset(node_ptr, 0, node_features.size() * sizeof(float));

        int current_atom_idx = 0;
        for (size_t i = 0; i < reactants.size(); ++i) {
            for (const auto& atom : reactants[i].atoms) {
                if (current_atom_idx >= config_.max_atoms) break;

                float* feat_ptr = node_ptr + (current_atom_idx * config_.atom_feature_dim);
                // Feature: One-hot atomic number
                if (atom.atomic_number > 0 && atom.atomic_number < config_.atom_feature_dim - 1) {
                    feat_ptr[atom.atomic_number - 1] = 1.0f;
                }
                // Feature: Which reactant this atom belongs to
                feat_ptr[config_.atom_feature_dim - 1] = (float)i;

                current_atom_idx++;
            }
        }

        // 3. Adjacency Matrix [1, MaxAtoms, MaxAtoms]
        adj_matrix.resize({1, (int64_t)config_.max_atoms, (int64_t)config_.max_atoms}, core::DataType::kFLOAT);
        float* adj_ptr = static_cast<float*>(adj_matrix.data());
        std::memset(adj_ptr, 0, adj_matrix.size() * sizeof(float));

        for (size_t i = 0; i < reactants.size(); ++i) {
            int offset = atom_offsets[i];
            for (const auto& bond : reactants[i].bonds) {
                int u = bond.atom_idx_1 + offset;
                int v = bond.atom_idx_2 + offset;
                if (u < config_.max_atoms && v < config_.max_atoms) {
                    adj_ptr[u * config_.max_atoms + v] = (float)bond.bond_type;
                    adj_ptr[v * config_.max_atoms + u] = (float)bond.bond_type; // Symmetric
                }
            }
        }
    }

    // --- Post-processing: Decode output graph ---
    ReactionResult decode_output() {
        ReactionResult result;

        // 1. Decode Yield (Assuming it's a separate scalar output)
        const float* yield_ptr = static_cast<const float*>(out_yield.data());
        result.yield_percent = yield_ptr[0]; // Assumes Sigmoid output [0,1]

        // 2. Decode Product Graph
        // Output Node Features [1, MaxAtoms, NumFeatures]
        // Output Adjacency Matrix [1, MaxAtoms, MaxAtoms]
        const float* out_node_ptr = static_cast<const float*>(out_node_features.data());
        const float* out_adj_ptr = static_cast<const float*>(out_adj_matrix.data());

        // A. Decode Atoms
        for (int i = 0; i < config_.max_atoms; ++i) {
            const float* feat = out_node_ptr + (i * config_.atom_feature_dim);
            // ArgMax to find atomic number
            int max_idx = 0;
            float max_val = feat[0];
            for (int j = 1; j < config_.atom_feature_dim; ++j) {
                if (feat[j] > max_val) {
                    max_val = feat[j];
                    max_idx = j;
                }
            }
            // If confidence is high and not "padding"
            if (max_val > 0.5 && max_idx > 0) {
                result.primary_product.atoms.push_back({max_idx + 1});
            }
        }

        // B. Decode Bonds
        // Threshold the adjacency matrix to find connections
        for (int i = 0; i < config_.max_atoms; ++i) {
            for (int j = i + 1; j < config_.max_atoms; ++j) { // Iterate upper triangle
                float bond_val = out_adj_ptr[i * config_.max_atoms + j];
                int bond_type = (int)std::round(bond_val);

                if (bond_type > 0) {
                    result.primary_product.bonds.push_back({i, j, bond_type});
                }
            }
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

ReactionForecaster::ReactionForecaster(const ForecasterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ReactionForecaster::~ReactionForecaster() = default;
ReactionForecaster::ReactionForecaster(ReactionForecaster&&) noexcept = default;
ReactionForecaster& ReactionForecaster::operator=(ReactionForecaster&&) noexcept = default;

ReactionResult ReactionForecaster::predict(const std::vector<MoleculeGraph>& reactants) {
    if (!pimpl_) throw std::runtime_error("ReactionForecaster is null.");

    // 1. Prepare Input
    pimpl_->prepare_input(reactants);

    // 2. Inference
    // Graph-to-Graph models typically have multiple outputs
    pimpl_->engine_->predict(
        {pimpl_->node_features, pimpl_->adj_matrix},
        {pimpl_->out_node_features, pimpl_->out_adj_matrix, pimpl_->out_yield}
    );

    // 3. Decode
    return pimpl_->decode_output();
}

} // namespace xinfer::zoo::chemistry