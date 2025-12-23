#pragma once
#include <xinfer/core/tensor.h>
#include "types.h"

namespace xinfer::preproc {

    class IGraphPreprocessor {
    public:
        virtual ~IGraphPreprocessor() = default;
        virtual void init(const graph::GraphConfig& config) = 0;

        /**
         * @brief Convert list of edges to Graph Tensors.
         *
         * @param edges Vector of interactions (src, dst, weight).
         * @param out_structure The structure tensor (Adjacency or Edge Index).
         * @param out_weights The edge attribute tensor.
         */
        virtual void process(const std::vector<graph::Edge>& edges,
                             core::Tensor& out_structure,
                             core::Tensor& out_weights) = 0;
    };

}