Here is the complete file structure and code for the **Text** (NLP) and **Tabular** (SIEM/Data) preprocessing modules.

This structure allows you to process high-throughput logs for **Blackbox SIEM** and text prompts for **xTorch** LLMs using the same uniform API as your image/audio pipelines.

### 📂 File Structure

```text
xinfer/
├── include/xinfer/preproc/
│   ├── text/
│   │   ├── text_preprocessor.h     # Interface (ITextPreproc)
│   │   └── types.h                 # Enums: TokenizerType, Vocab, Padding
│   └── tabular/
│       ├── tabular_preprocessor.h  # Interface (ITabularPreproc)
│       └── types.h                 # Enums: ColumnType, Encoding, Schema
│
├── src/preproc/
│   ├── text/
│   │   ├── cpu/
│   │   │   ├── bert_tokenizer.cpp  # WordPiece algorithm (CPU)
│   │   │   ├── bpe_tokenizer.cpp   # Byte-Pair Encoding (GPT/Llama)
│   │   │   └── string_utils.cpp    # Unicode normalization helpers
│   │   └── cuda/                   # [Optional High-Perf]
│   │       └── cubert_tokenizer.cu # NVIDIA cuBERT implementation
│   │
│   └── tabular/
│       ├── cpu/
│       │   ├── log_encoder.cpp     # Main CPU implementation
│       │   ├── ip_utils.cpp        # Optimized IPv4/IPv6 -> Float conversions
│       │   └── scalers.cpp         # Math for MinMax/Standard scaling
│       └── cuda/                   # [Future SIEM Upgrade]
│           └── cudf_wrapper.cu     # GPU DataFrame logic (RAPIDS style)
```

---

### 1. Tabular Preprocessing (For Blackbox SIEM)

This module converts security logs (IPs, Usernames, Timestamps) into normalized tensors for Anomaly Detection.

**File:** `include/xinfer/preproc/tabular/types.h`

```cpp
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
```

**File:** `include/xinfer/preproc/tabular/tabular_preprocessor.h`

```cpp
#pragma once
#include <xinfer/core/tensor.h>
#include "types.h"

namespace xinfer::preproc {

class ITabularPreprocessor {
public:
    virtual ~ITabularPreprocessor() = default;

    /**
     * @brief Initialize with the dataset schema.
     * Contains the means/stds/categories learned during training.
     */
    virtual void init(const std::vector<tabular::ColumnSchema>& schema) = 0;

    /**
     * @brief Process a single log row into a Tensor.
     * 
     * @param raw_row Vector of strings (e.g. ["TCP", "192.168.1.5", "80", "Error"])
     * @param dst Output tensor (Float32). Size = sum of encoded feature widths.
     */
    virtual void process(const tabular::TableRow& raw_row, core::Tensor& dst) = 0;

    /**
     * @brief Batch Process (Optimized for SIEM Throughput).
     */
    virtual void process_batch(const std::vector<tabular::TableRow>& rows, core::Tensor& dst) = 0;
};

}
```

**File:** `src/preproc/tabular/cpu/log_encoder.cpp` (Basic Implementation)

```cpp
#include <xinfer/preproc/tabular/tabular_preprocessor.h>
#include <xinfer/core/logging.h>
#include <cmath>
#include <sstream>

namespace xinfer::preproc {

class CpuTabularPreprocessor : public ITabularPreprocessor {
public:
    void init(const std::vector<tabular::ColumnSchema>& schema) override {
        m_schema = schema;
        // Calculate total output size per row
        m_total_features = 0;
        for(const auto& col : m_schema) {
            if (col.type == tabular::ColumnType::IP_ADDRESS) m_total_features += 4; // 4 octets
            else if (col.type != tabular::ColumnType::IGNORE) m_total_features += 1;
        }
    }

    void process(const tabular::TableRow& row, core::Tensor& dst) override {
        if (row.size() != m_schema.size()) {
            XINFER_LOG_ERROR("Row size mismatch schema.");
            return;
        }

        dst.resize({1, (int64_t)m_total_features}, core::DataType::kFLOAT);
        float* ptr = static_cast<float*>(dst.data());
        int idx = 0;

        for (size_t i = 0; i < row.size(); ++i) {
            const auto& col = m_schema[i];
            const std::string& val = row[i];

            if (col.type == tabular::ColumnType::NUMERICAL) {
                float fval = std::stof(val);
                if (col.encoding == tabular::EncodingType::STANDARD_SCALE) {
                    ptr[idx++] = (fval - col.mean) / col.std;
                } else {
                    ptr[idx++] = fval;
                }
            } 
            else if (col.type == tabular::ColumnType::CATEGORICAL) {
                if (col.category_map.count(val)) {
                    ptr[idx++] = col.category_map.at(val);
                } else {
                    ptr[idx++] = col.unknown_value;
                }
            }
            else if (col.type == tabular::ColumnType::IP_ADDRESS) {
                // Parse IP "192.168.1.1" -> 4 floats normalized 0-1
                int a, b, c, d;
                char dot;
                std::stringstream ss(val);
                ss >> a >> dot >> b >> dot >> c >> dot >> d;
                ptr[idx++] = a / 255.0f;
                ptr[idx++] = b / 255.0f;
                ptr[idx++] = c / 255.0f;
                ptr[idx++] = d / 255.0f;
            }
        }
    }

    void process_batch(const std::vector<tabular::TableRow>& rows, core::Tensor& dst) override {
        // Implementation: Loop over rows and fill dst (Batch x Features)
        // Omitted for brevity
    }

private:
    std::vector<tabular::ColumnSchema> m_schema;
    size_t m_total_features = 0;
};

}
```

---

### 2. Text Preprocessing (For NLP/LLMs)

This module converts natural language into Token IDs for Transformers.

**File:** `include/xinfer/preproc/text/types.h`

```cpp
#pragma once
#include <string>
#include <map>

namespace xinfer::preproc::text {

enum class TokenizerType {
    BERT_WORDPIECE = 0, // BERT, DistilBERT
    GPT_BPE = 1,        // GPT-2, RoBERTa
    SENTENCEPIECE = 2,  // Llama, T5
    WHITESPACE = 3
};

struct TokenizerConfig {
    std::string vocab_path;      // Path to vocab.txt
    std::string merges_path;     // Path to merges.txt (for BPE)
    int max_length = 512;        // Context window
    bool do_lower_case = true;
    bool add_special_tokens = true; // [CLS], [SEP] or <s>, </s>
    
    // Special Token IDs
    int pad_token_id = 0;
    int unk_token_id = 100;
    int cls_token_id = 101;
    int sep_token_id = 102;
};

}
```

**File:** `include/xinfer/preproc/text/text_preprocessor.h`

```cpp
#pragma once
#include <xinfer/core/tensor.h>
#include "types.h"

namespace xinfer::preproc {

class ITextPreprocessor {
public:
    virtual ~ITextPreprocessor() = default;

    virtual void init(const text::TokenizerConfig& config) = 0;

    /**
     * @brief Tokenize a single string.
     * 
     * @param text Input string.
     * @param input_ids Output tensor [1, max_len] (INT32/INT64).
     * @param attention_mask Output tensor [1, max_len] (INT32).
     */
    virtual void process(const std::string& text, 
                         core::Tensor& input_ids, 
                         core::Tensor& attention_mask) = 0;

    /**
     * @brief Decode IDs back to string (for Generative AI output).
     */
    virtual std::string decode(const core::Tensor& output_ids) = 0;
};

}
```

**File:** `src/preproc/text/cpu/bert_tokenizer.cpp` (Partial Logic)

```cpp
#include <xinfer/preproc/text/text_preprocessor.h>
#include <xinfer/core/logging.h>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace xinfer::preproc {

class BertTokenizer : public ITextPreprocessor {
public:
    void init(const text::TokenizerConfig& config) override {
        m_config = config;
        load_vocab(config.vocab_path);
    }

    void process(const std::string& text, 
                 core::Tensor& input_ids, 
                 core::Tensor& attention_mask) override {
        
        // 1. Normalize (Lowercase)
        std::string processed_text = text;
        if (m_config.do_lower_case) {
            std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(), ::tolower);
        }

        // 2. Split by whitespace (Basic)
        std::vector<std::string> words;
        std::stringstream ss(processed_text);
        std::string item;
        while (getline(ss, item, ' ')) words.push_back(item);

        // 3. WordPiece Tokenization (Simplified)
        std::vector<int> tokens;
        if (m_config.add_special_tokens) tokens.push_back(m_config.cls_token_id);

        for (const auto& word : words) {
            // Check full word
            if (m_vocab.count(word)) {
                tokens.push_back(m_vocab[word]);
            } else {
                // Real implementation handles subwords (play, ##ing)
                tokens.push_back(m_config.unk_token_id);
            }
            if (tokens.size() >= m_config.max_length - 1) break;
        }

        if (m_config.add_special_tokens) tokens.push_back(m_config.sep_token_id);

        // 4. Create Tensors
        input_ids.resize({1, (int64_t)m_config.max_length}, core::DataType::kINT32);
        attention_mask.resize({1, (int64_t)m_config.max_length}, core::DataType::kINT32);

        int32_t* ids_ptr = static_cast<int32_t*>(input_ids.data());
        int32_t* mask_ptr = static_cast<int32_t*>(attention_mask.data());

        // Fill data + padding
        for (int i = 0; i < m_config.max_length; ++i) {
            if (i < tokens.size()) {
                ids_ptr[i] = tokens[i];
                mask_ptr[i] = 1;
            } else {
                ids_ptr[i] = m_config.pad_token_id;
                mask_ptr[i] = 0;
            }
        }
    }

    std::string decode(const core::Tensor& output_ids) override {
        return ""; // Implementation omitted
    }

private:
    text::TokenizerConfig m_config;
    std::map<std::string, int> m_vocab;

    void load_vocab(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        int id = 0;
        while (std::getline(file, line)) {
            m_vocab[line] = id++;
        }
    }
};

}
```