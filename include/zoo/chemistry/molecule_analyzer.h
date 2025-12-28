#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::chemistry {

    /**
     * @brief A single atom in the molecule.
     */
    struct Atom {
        int atomic_number; // e.g., 6 for Carbon, 8 for Oxygen
        // Other features: charge, hybridization, etc.
    };

    /**
     * @brief A bond between two atoms.
     */
    struct Bond {
        int atom_idx_1;
        int atom_idx_2;
        int bond_type; // 1=Single, 2=Double, 3=Triple
    };

    /**
     * @brief The molecule represented as a graph.
     */
    struct MoleculeGraph {
        std::vector<Atom> atoms;
        std::vector<Bond> bonds;
    };

    /**
     * @brief Predicted properties.
     */
    struct MoleculeProperties {
        float solubility;    // e.g., logS
        float toxicity;      // e.g., LD50 prediction
        float binding_affinity; // for drug discovery

        // Map of all raw model outputs
        std::map<std::string, float> raw_outputs;
    };

    struct AnalyzerConfig {
        // Hardware Target (GNNs run well on GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., gcn_lipophilicity.engine)
        std::string model_path;

        // --- Model Specs ---
        // Max atoms/nodes the model can handle
        int max_nodes = 128;
        // Number of features per atom (e.g., atomic number, charge, etc.)
        int atom_feature_dim = 9;

        // Output property names, in the order the model returns them
        std::vector<std::string> property_names = {"Solubility"};

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class MoleculeAnalyzer {
    public:
        explicit MoleculeAnalyzer(const AnalyzerConfig& config);
        ~MoleculeAnalyzer();

        // Move semantics
        MoleculeAnalyzer(MoleculeAnalyzer&&) noexcept;
        MoleculeAnalyzer& operator=(MoleculeAnalyzer&&) noexcept;
        MoleculeAnalyzer(const MoleculeAnalyzer&) = delete;
        MoleculeAnalyzer& operator=(const MoleculeAnalyzer&) = delete;

        /**
         * @brief Predict properties for a molecule graph.
         *
         * Pipeline:
         * 1. Featurization: Convert Atom/Bond info to Tensors.
         * 2. Inference: Run GNN.
         * 3. Postprocess: Map output vector to properties.
         *
         * @param molecule The molecular structure.
         * @return Predicted properties.
         */
        MoleculeProperties analyze(const MoleculeGraph& molecule);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::chemistry