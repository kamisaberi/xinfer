#include <xinfer/preproc/graph/graph_preprocessor.h>
#include <xinfer/core/logging.h>
#include <algorithm>
#include <cstring>

namespace xinfer::preproc {

class CpuGraphBuilder : public IGraphPreprocessor {
public:
    void init(const graph::GraphConfig& config) override {
        m_config = config;
    }

    void process(const std::vector<graph::Edge>& edges,
                 core::Tensor& out_structure,
                 core::Tensor& out_weights) override {

        if (m_config.output_format == graph::GraphFormat::ADJACENCY_MATRIX) {
            // Dense Matrix [N, N]
            int N = m_config.max_nodes;
            out_structure.resize({1, (int64_t)N, (int64_t)N}, core::DataType::kFLOAT);
            float* adj = static_cast<float*>(out_structure.data());
            std::fill(adj, adj + (N * N), 0.0f); // Clear

            // Weights are embedded in Adjacency for Dense mode
            // (out_weights is unused or duplicate)

            for (const auto& e : edges) {
                if (e.source_id < N && e.target_id < N) {
                    adj[e.source_id * N + e.target_id] = e.weight;
                    if (!m_config.directed) {
                        adj[e.target_id * N + e.source_id] = e.weight;
                    }
                }
            }

            // Normalize (Row Normalization D^-1 * A)
            if (m_config.normalize) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) sum += adj[i * N + j];
                    if (sum > 1e-5f) {
                        for (int j = 0; j < N; ++j) adj[i * N + j] /= sum;
                    }
                }
            }
        }
        else if (m_config.output_format == graph::GraphFormat::COO_SPARSE) {
            // Coordinate List (Edge Index) usually 2 rows: [Source...], [Target...]
            // Shape: [2, NumEdges]
            int E = edges.size();
            out_structure.resize({2, (int64_t)E}, core::DataType::kINT64);
            out_weights.resize({(int64_t)E}, core::DataType::kFLOAT);

            int64_t* idx_ptr = static_cast<int64_t*>(out_structure.data());
            float* w_ptr = static_cast<float*>(out_weights.data());

            for (int i = 0; i < E; ++i) {
                // Row 0: Sources
                idx_ptr[i] = edges[i].source_id;
                // Row 1: Targets (offset by E)
                idx_ptr[E + i] = edges[i].target_id;

                w_ptr[i] = edges[i].weight;
            }
        }
    }

private:
    graph::GraphConfig m_config;
};

}