#include <xinfer/zoo/chemistry/molecule_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Preproc/Postproc is custom numerical logic for graphs.

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

namespace xinfer::zoo::chemistry {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct MoleculeAnalyzer::Impl {
    AnalyzerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor node_features;
    core::Tensor edge_index;
    // Optional: core::Tensor edge_attr;

    core::Tensor output_tensor;

    Impl(const AnalyzerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("MoleculeAnalyzer: Failed to load model.");
        }
    }

    // --- Core Logic: Graph Featurization ---
    void prepare_input(const MoleculeGraph& mol) {
        int num_atoms = std::min((int)mol.atoms.size(), config_.max_nodes);
        int num_bonds = mol.bonds.size();

        // 1. Node Features Tensor [NumAtoms, NumFeatures]
        // (Padded to MaxNodes)
        node_features.resize({(int64_t)config_.max_nodes, (int64_t)config_.atom_feature_dim}, core::DataType::kFLOAT);
        float* node_ptr = static_cast<float*>(node_features.data());
        std::memset(node_ptr, 0, node_features.size() * sizeof(float));

        for (int i = 0; i < num_atoms; ++i) {
            float* atom_feats = node_ptr + (i * config_.atom_feature_dim);

            // Simple One-Hot encoding of atomic number (or use embedding layer)
            // This is a simplified feature set. Real models use more complex features.
            int atomic_num = mol.atoms[i].atomic_number;
            if (atomic_num > 0 && atomic_num < config_.atom_feature_dim) {
                atom_feats[atomic_num - 1] = 1.0f; // e.g. H=1 -> idx 0, C=6 -> idx 5
            }
        }

        // 2. Edge Index Tensor [2, NumBonds * 2]
        // GNNs need directed edges, so each bond becomes two edges (A->B, B->A)
        int num_edges = num_bonds * 2;
        edge_index.resize({2, (int64_t)num_edges}, core::DataType::kINT64);
        int64_t* edge_ptr = static_cast<int64_t*>(edge_index.data());

        int edge_count = 0;
        for (const auto& bond : mol.bonds) {
            // Edge 1
            edge_ptr[edge_count] = bond.atom_idx_1; // Source
            edge_ptr[edge_count + num_edges] = bond.atom_idx_2; // Target
            edge_count++;

            // Edge 2 (Reverse)
            edge_ptr[edge_count] = bond.atom_idx_2; // Source
            edge_ptr[edge_count + num_edges] = bond.atom_idx_1; // Target
            edge_count++;
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

MoleculeAnalyzer::MoleculeAnalyzer(const AnalyzerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

MoleculeAnalyzer::~MoleculeAnalyzer() = default;
MoleculeAnalyzer::MoleculeAnalyzer(MoleculeAnalyzer&&) noexcept = default;
MoleculeAnalyzer& MoleculeAnalyzer::operator=(MoleculeAnalyzer&&) noexcept = default;

MoleculeProperties MoleculeAnalyzer::analyze(const MoleculeGraph& molecule) {
    if (!pimpl_) throw std::runtime_error("MoleculeAnalyzer is null.");

    // 1. Prepare Input Tensors
    pimpl_->prepare_input(molecule);

    // 2. Inference
    // GNN models expect specific input names (e.g., 'x', 'edge_index')
    // The backend should handle this, but here we pass them in order.
    pimpl_->engine_->predict({pimpl_->node_features, pimpl_->edge_index}, {pimpl_->output_tensor});

    // 3. Post-process
    MoleculeProperties props;

    // Output is typically [1, NumProperties]
    const float* out_ptr = static_cast<const float*>(pimpl_->output_tensor.data());

    // Map output vector to named properties
    for (size_t i = 0; i < pimpl_->config_.property_names.size(); ++i) {
        std::string name = pimpl_->config_.property_names[i];
        float value = out_ptr[i];

        props.raw_outputs[name] = value;

        // Simple mapping for primary properties
        if (name == "Solubility") props.solubility = value;
        else if (name == "Toxicity") props.toxicity = value;
        else if (name == "Binding") props.binding_affinity = value;
    }

    return props;
}

} // namespace xinfer::zoo::chemistry