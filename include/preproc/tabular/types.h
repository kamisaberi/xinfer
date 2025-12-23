#pragma once
#include <string>
#include <vector>
#include <map>
#include <variant>

namespace xinfer::preproc::tabular {

    // What kind of data is in this column?
    enum class ColumnType {
        NUMERICAL = 0,   // Packet Size, CPU Usage
        CATEGORICAL = 1, // Protocol (TCP/UDP), Country, Username
        IP_ADDRESS = 2,  // 192.168.1.1 (Needs special splitting)
        TIMESTAMP = 3,   // UNIX Timestamp (needs cyclic encoding or normalization)
        IGNORE = 4       // Skip this column
    };

    // How should we encode it?
    enum class EncodingType {
        NONE = 0,           // Passthrough
        STANDARD_SCALE = 1, // (x - mean) / std
        MIN_MAX_SCALE = 2,  // (x - min) / (max - min)
        LABEL_ENCODE = 3,   // String -> Int ID
        ONE_HOT = 4,        // String -> Vector of 0s and 1s
        IP_SPLIT_NORM = 5   // 192.168.1.1 -> [0.75, 0.66, 0.003, 0.003]
    };

    struct ColumnSchema {
        std::string name;
        ColumnType type;
        EncodingType encoding;

        // Stats for scaling (loaded from training phase)
        float mean = 0.0f;
        float std = 1.0f;
        float min = 0.0f;
        float max = 1.0f;

        // Map for Categorical data
        std::map<std::string, float> category_map;
        float unknown_value = -1.0f;
    };

    // A single row of raw data (parsed from JSON or CSV)
    using TableRow = std::vector<std::string>;

}