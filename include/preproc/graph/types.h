#pragma once
#include <vector>
#include <string>

namespace xinfer::preproc::graph {

    struct Edge {
        int source_id;
        int target_id;
        float weight = 1.0f; // e.g., Bytes sent, Frequency
    };

    enum class GraphFormat {
        ADJACENCY_MATRIX = 0, // Dense [N, N] (Good for small graphs)
        COO_SPARSE = 1,       // Coordinate List (2, E) (Standard for PyTorch Geometric)
        CSR_SPARSE = 2        // Compressed Sparse Row (Fastest for matrix mul)
    };

    struct GraphConfig {
        int max_nodes;        // Maximum nodes in the graph window
        bool directed = true;
        bool normalize = true; // Normalize weights (e.g., Degree Normalization)
        GraphFormat output_format = GraphFormat::COO_SPARSE;
    };

}