#include <iostream>
#include <vector>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/graph/graph_preprocessor.h> // The Graph Builder
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    // Target: NVIDIA GPU (Sparse Matrix multiplication is fast on CUDA)
    Target target = Target::NVIDIA_TRT;

    // 1. Setup Engine (GCN / GraphSAGE model)
    auto engine = backends::BackendFactory::create(target);
    engine->load_model("siem_gcn.engine");

    // 2. Setup Graph Preprocessor
    // We need to implement a Factory for this, or use the class directly if CPU-only
    auto graph_prep = std::make_unique<preproc::CpuGraphBuilder>();

    preproc::graph::GraphConfig g_cfg;
    g_cfg.max_nodes = 1000; // Window size of unique IPs
    g_cfg.output_format = preproc::graph::GraphFormat::COO_SPARSE; // Edge Index [2, E]
    graph_prep->init(g_cfg);

    // 3. Simulated Log Ingestion (SrcIP -> DstIP)
    std::vector<preproc::graph::Edge> edges = {
        {0, 1, 500},  // Node 0 connected to 1 (500 bytes)
        {1, 2, 1200}, // Node 1 connected to 2 (Lateral move?)
        {2, 0, 40},   // Back to 0
        // ... hundreds more ...
    };

    // 4. Build Graph Tensors
    core::Tensor edge_index; // Structure
    core::Tensor edge_attr;  // Weights
    graph_prep->process(edges, edge_index, edge_attr);

    // 5. Node Features (e.g., embeddings of IP addresses)
    // Usually pre-calculated or learned. Mocking here:
    core::Tensor node_features({1, 1000, 64}, core::DataType::kFLOAT);

    // 6. Inference
    core::Tensor anomaly_scores;
    // GNN inputs: [NodeFeatures, EdgeIndex, EdgeAttr]
    engine->predict({node_features, edge_index, edge_attr}, {anomaly_scores});

    // 7. Check for Anomalies
    const float* scores = (float*)anomaly_scores.data();
    for(int i=0; i<edges.size(); ++i) {
        if (scores[i] > 0.8f) {
            std::cout << "[ALERT] Anomalous Edge: "
                      << edges[i].source_id << " -> " << edges[i].target_id << std::endl;
        }
    }

    return 0;
}