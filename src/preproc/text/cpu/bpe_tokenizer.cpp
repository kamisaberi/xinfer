#include "bpe_tokenizer.h"
#include <xinfer/core/logging.h>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <regex>
#include <limits>

namespace xinfer::preproc::text {

BpeTokenizer::BpeTokenizer() {}
BpeTokenizer::~BpeTokenizer() {}

void BpeTokenizer::init(const TokenizerConfig& config) {
    m_config = config;
    build_byte_encoder();
    load_vocab(config.vocab_path);
    load_merges(config.merges_path);
}

// =================================================================================
// 1. Initialization & Loading
// =================================================================================

void BpeTokenizer::build_byte_encoder() {
    // GPT-2 BPE Byte Encoder logic:
    // Maps bytes 0-255 to unicode characters to avoid UNK.
    // Standard printable ASCII is kept as is.
    // Others are mapped to Latin-1 Supplement characters (starting U+0100).

    std::vector<int> bs;
    // ASCII range
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n); // Map to U+0100 range
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        // Convert int unicode to UTF-8 string
        unsigned char byte_val = static_cast<unsigned char>(bs[i]);
        int code_point = cs[i];

        std::string utf8_char;
        if (code_point < 0x80) {
            utf8_char += (char)code_point;
        } else if (code_point < 0x800) {
            utf8_char += (char)((code_point >> 6) | 0xC0);
            utf8_char += (char)((code_point & 0x3F) | 0x80);
        } else {
            // Simplified for the GPT-2 range (doesn't go above 0xFFFF usually here)
            utf8_char += (char)((code_point >> 12) | 0xE0);
            utf8_char += (char)(((code_point >> 6) & 0x3F) | 0x80);
            utf8_char += (char)((code_point & 0x3F) | 0x80);
        }

        m_byte_encoder[byte_val] = utf8_char;
        m_byte_decoder[utf8_char] = byte_val;
    }
}

void BpeTokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open vocab file: " + path);
        return;
    }

    // Lightweight JSON parser logic (Token -> ID)
    // Assumes standard {"token": id, ...} format
    std::string line;
    while (std::getline(file, line)) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        size_t quote_start = line.find('"');
        size_t quote_end = line.rfind('"', colon_pos);
        if (quote_start == std::string::npos || quote_end == std::string::npos) continue;

        std::string token = line.substr(quote_start + 1, quote_end - quote_start - 1);

        // Unescape unicode (basic handling)
        // In real JSON files, unicode chars might be \u0120.
        // For simplicity, assuming file is UTF-8 or standard formatted.

        std::string id_str = line.substr(colon_pos + 1);
        // Remove trailing comma if present
        if (id_str.find(',') != std::string::npos) id_str.pop_back();

        try {
            int id = std::stoi(id_str);
            m_vocab[token] = id;
            m_decoder[id] = token;
        } catch (...) {}
    }
}

void BpeTokenizer::load_merges(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open merges file: " + path);
        return;
    }

    std::string line;
    std::getline(file, line); // Skip version line usually first

    int rank = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t space = line.find(' ');
        if (space == std::string::npos) continue;

        std::string first = line.substr(0, space);
        std::string second = line.substr(space + 1);
        // Trim newline
        if (!second.empty() && second.back() == '\r') second.pop_back();

        m_bpe_ranks[{first, second}] = rank++;
    }
}

// =================================================================================
// 2. BPE Algorithm Logic
// =================================================================================

std::vector<std::string> BpeTokenizer::get_pairs(const std::vector<std::string>& word) {
    std::vector<std::string> pairs;
    if (word.size() < 2) return pairs;

    // We actually need set of pairs, but return distinct list for iteration
    // Using string serialization for pair key simply
    return pairs; // Unused, logic in bpe() directly
}

std::vector<std::string> BpeTokenizer::bpe(const std::string& token) {
    if (m_cache.count(token)) return m_cache[token];

    // Split word into individual characters (utf-8 aware strings)
    std::vector<std::string> word;
    for (size_t i = 0; i < token.length(); ) {
        // Simple UTF-8 char len detection
        char c = token[i];
        int len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        word.push_back(token.substr(i, len));
        i += len;
    }

    while (word.size() > 1) {
        int min_rank = std::numeric_limits<int>::max();
        std::pair<std::string, std::string> best_pair;
        bool found = false;

        // Find the lowest ranked pair adjacent in the word
        for (size_t i = 0; i < word.size() - 1; ++i) {
            std::pair<std::string, std::string> pair = {word[i], word[i+1]};
            if (m_bpe_ranks.count(pair)) {
                int rank = m_bpe_ranks[pair];
                if (rank < min_rank) {
                    min_rank = rank;
                    best_pair = pair;
                    found = true;
                }
            }
        }

        if (!found) break; // No more merges possible

        // Apply Merge
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 &&
                word[i] == best_pair.first &&
                word[i+1] == best_pair.second) {

                new_word.push_back(word[i] + word[i+1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = new_word;
    }

    m_cache[token] = word;
    return word;
}

// =================================================================================
// 3. Main Process
// =================================================================================

void BpeTokenizer::process(const std::string& text,
                           core::Tensor& input_ids,
                           core::Tensor& attention_mask) {

    // 1. Pre-tokenize (Regex split)
    // GPT-2 Regex: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // Approximating with simple whitespace handling for C++ standard lib

    std::vector<int> bpe_tokens;

    // Add Start Token (BOS) if configured (e.g. for Llama)
    // if (m_config.add_special_tokens) bpe_tokens.push_back(1);

    // Crude whitespace splitting (replace with regex in production)
    std::regex re(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+)");
    std::sregex_iterator it(text.begin(), text.end(), re);
    std::sregex_iterator end;

    for (; it != end; ++it) {
        std::string raw_token = it->str();

        // 2. Byte Encode (Map chars to the vocab-safe unicode chars)
        std::string encoded_token = "";
        for (char c : raw_token) {
            encoded_token += m_byte_encoder[static_cast<unsigned char>(c)];
        }

        // 3. Apply BPE
        std::vector<std::string> subwords = bpe(encoded_token);

        // 4. Map to IDs
        for (const auto& sub : subwords) {
            if (m_vocab.count(sub)) {
                bpe_tokens.push_back(m_vocab[sub]);
            } else {
                bpe_tokens.push_back(m_config.unk_token_id);
            }
        }
    }

    // 5. Tensor Creation
    input_ids.resize({1, (int64_t)m_config.max_length}, core::DataType::kINT32);
    attention_mask.resize({1, (int64_t)m_config.max_length}, core::DataType::kINT32);

    int32_t* ids_ptr = static_cast<int32_t*>(input_ids.data());
    int32_t* mask_ptr = static_cast<int32_t*>(attention_mask.data());

    for (int i = 0; i < m_config.max_length; ++i) {
        if (i < bpe_tokens.size()) {
            ids_ptr[i] = bpe_tokens[i];
            mask_ptr[i] = 1;
        } else {
            ids_ptr[i] = m_config.pad_token_id;
            mask_ptr[i] = 0;
        }
    }
}

// =================================================================================
// 4. Decode
// =================================================================================

std::string BpeTokenizer::decode(const core::Tensor& output_ids) {
    std::string text = "";
    const int32_t* ids = static_cast<const int32_t*>(output_ids.data());
    size_t count = output_ids.size();

    for (size_t i = 0; i < count; ++i) {
        int id = ids[i];
        if (m_decoder.count(id)) {
            text += m_decoder[id];
        }
    }

    // Reverse Byte Encoding (Map unicode chars back to bytes)
    std::string decoded = "";
    for (size_t i = 0; i < text.length(); ) {
        // Greedy match for UTF-8 chars in decoder map
        // Since m_byte_decoder keys are strings (utf8 chars), we need to match them.
        // Simplification: We know mapped chars are either 1, 2, or 3 bytes.

        bool matched = false;
        for (int len = 3; len >= 1; --len) {
            if (i + len > text.length()) continue;
            std::string sub = text.substr(i, len);
            if (m_byte_decoder.count(sub)) {
                decoded += (char)m_byte_decoder[sub];
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Should not happen if vocab is correct
            i++;
        }
    }

    return decoded;
}

} // namespace xinfer::preproc::text