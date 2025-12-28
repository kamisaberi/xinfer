#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::chemistry {

    // --- We reuse the MoleculeGraph struct from the MoleculeAnalyzer ---
    struct Atom { int atomic_number; };
    struct Bond { int atom_idx_1; int atom_idx_2; int bond_type; };
    struct MoleculeGraph {
        std::vector<Atom> atoms;
        std::vector<Bond> bonds;
    };
    // ---

    /**
     * @brief The predicted outcome of the reaction.
     */
    struct ReactionResult {
        // Primary product graph
        MoleculeGraph primary_product;

        // Predicted yield of the primary product (0.0 to 1.0)
        float yield_percent;

        // List of likely side products
        std::vector<MoleculeGraph> side_products;
    };

    struct ForecasterConfig {
        // Hardware Target (GNNs are best on GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., chemformer_reaction.engine)
        // A Graph-to-Graph Transformer or similar model.
        std::string model_path;

        // Model Specs
        int max_atoms = 256;
        int atom_feature_dim = 9;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ReactionForecaster {
    public:
        explicit ReactionForecaster(const ForecasterConfig& config);
        ~ReactionForecaster();

        // Move semantics
        ReactionForecaster(ReactionForecaster&&) noexcept;
        ReactionForecaster& operator=(ReactionForecaster&&) noexcept;
        ReactionForecaster(const ReactionForecaster&) = delete;
        ReactionForecaster& operator=(const ReactionForecaster&) = delete;

        /**
         * @brief Predict the outcome of a reaction.
         *
         * @param reactants A list of input molecule graphs.
         * @return The predicted product(s) and yield.
         */
        ReactionResult predict(const std::vector<MoleculeGraph>& reactants);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::chemistry