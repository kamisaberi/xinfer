#include "bert_tokenizer.h"
#include <xinfer/core/logging.h>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace xinfer::preproc::text {

BertTokenizer::BertTokenizer() {}
BertTokenizer::~BertTokenizer() {}

void BertTokenizer::init(const TokenizerConfig& config) {
    m_config = config;
    load_vocab(config.vocab_path);
}

void BertTokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open vocab file: " + path);
        return;
    }

    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        // Trim newline chars if present
        if (!line.empty() && line.back() == '\r') line.pop_back();

        m_vocab[line] = id;
        m_ids_to_tokens[id] = line;
        id++;
    }
}

// Helper: Check if char is punctuation (simplified for C++)
static bool is_punctuation(char c) {
    return std::ispunct(static_cast<unsigned char>(c));
}

std::vector<std::string> BertTokenizer::basic_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current_token;

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        }
        else if (is_punctuation(c)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(std::string(1, c)); // Add punct as separate token
        }
        else {
            current_token += (m_config.do_lower_case ? std::tolower(c) : c);
        }
    }
    if (!current_token.empty()) tokens.push_back(current_token);

    return tokens;
}

std::vector<int> BertTokenizer::wordpiece_tokenize(const std::string& word) {
    std::vector<int> subword_ids;

    if (word.length() > 100) {
        subword_ids.push_back(m_config.unk_token_id);
        return subword_ids;
    }

    bool is_bad = false;
    int start = 0;
    int len = word.length();

    while (start < len) {
        int end = len;
        std::string cur_substr;
        bool found = false;

        // Greedy Longest-Match-First Strategy
        while (start < end) {
            std::string sub = word.substr(start, end - start);
            if (start > 0) sub = "##" + sub; // Add WordPiece prefix

            if (m_vocab.find(sub) != m_vocab.end()) {
                cur_substr = sub;
                found = true;
                break;
            }
            end--;
        }

        if (!found) {
            is_bad = true;
            break;
        }

        subword_ids.push_back(m_vocab[cur_substr]);
        start = end;
    }

    if (is_bad) {
        return { m_config.unk_token_id };
    }

    return subword_ids;
}

void BertTokenizer::process(const std::string& text,
                            core::Tensor& input_ids,
                            core::Tensor& attention_mask) {

    // 1. Basic Tokenize (Whitespace/Punctuation split)
    std::vector<std::string> basic_tokens = basic_tokenize(text);

    // 2. WordPiece Tokenize
    std::vector<int> final_ids;

    // Add [CLS]
    if (m_config.add_special_tokens) final_ids.push_back(m_config.cls_token_id);

    for (const auto& token : basic_tokens) {
        std::vector<int> pieces = wordpiece_tokenize(token);
        final_ids.insert(final_ids.end(), pieces.begin(), pieces.end());

        // Truncate if too long (reserve 1 slot for SEP)
        if (final_ids.size() >= m_config.max_length - 1) break;
    }

    // Add [SEP]
    if (m_config.add_special_tokens) final_ids.push_back(m_config.sep_token_id);

    // 3. Prepare Tensors
    input_ids.resize({1, (int64_t)m_config.max_length}, core::DataType::kINT32);
    attention_mask.resize({1, (int64_t)m_config.max_length}, core::DataType::kINT32);

    int32_t* ids_ptr = static_cast<int32_t*>(input_ids.data());
    int32_t* mask_ptr = static_cast<int32_t*>(attention_mask.data());

    // 4. Fill and Pad
    for (int i = 0; i < m_config.max_length; ++i) {
        if (i < final_ids.size()) {
            ids_ptr[i] = final_ids[i];
            mask_ptr[i] = 1; // Attention Mask = 1 for real tokens
        } else {
            ids_ptr[i] = m_config.pad_token_id;
            mask_ptr[i] = 0; // Attention Mask = 0 for padding
        }
    }
}

std::string BertTokenizer::decode(const core::Tensor& output_ids) {
    std::stringstream ss;
    const int32_t* ids = static_cast<const int32_t*>(output_ids.data());
    size_t count = output_ids.size();

    for (size_t i = 0; i < count; ++i) {
        int id = ids[i];

        // Skip special tokens
        if (id == m_config.pad_token_id ||
            id == m_config.cls_token_id ||
            id == m_config.sep_token_id) continue;

        if (m_ids_to_tokens.count(id)) {
            std::string token = m_ids_to_tokens[id];

            // Handle subwords "##ing"
            if (token.size() > 2 && token.substr(0, 2) == "##") {
                ss << token.substr(2); // Append without space
            } else {
                if (i > 0) ss << " "; // Add space before new word
                ss << token;
            }
        }
    }
    return ss.str();
}

} // namespace xinfer::preproc::text