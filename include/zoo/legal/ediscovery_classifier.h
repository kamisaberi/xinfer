#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::legal {

    /**
     * @brief A single classification tag for a document.
     */
    struct DocumentTag {
        std::string tag;        // e.g., "Relevant", "Privileged", "Hot_Doc"
        float confidence;
    };

    /**
     * @brief Result of the classification.
     */
    struct EDiscoveryResult {
        std::string document_id;
        std::vector<DocumentTag> tags;
        bool is_responsive; // Shorthand for "Relevant" tag found
    };

    struct EDiscoveryConfig {
        // Hardware Target (Batch processing runs well on GPU/NPU)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., bert_ediscovery_multilabel.onnx)
        // Expected Output: [Batch, NumLabels] Logits
        std::string model_path;

        // Tokenizer
        std::string vocab_path;
        int max_sequence_length = 512;

        // Label Map (Path to labels.txt: "Relevant", "Privileged", etc.)
        std::string labels_path;

        // Sensitivity
        float classification_threshold = 0.5f; // Threshold to apply Sigmoid outputs

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class EDiscoveryClassifier {
    public:
        explicit EDiscoveryClassifier(const EDiscoveryConfig& config);
        ~EDiscoveryClassifier();

        // Move semantics
        EDiscoveryClassifier(EDiscoveryClassifier&&) noexcept;
        EDiscoveryClassifier& operator=(EDiscoveryClassifier&&) noexcept;
        EDiscoveryClassifier(const EDiscoveryClassifier&) = delete;
        EDiscoveryClassifier& operator=(const EDiscoveryClassifier&) = delete;

        /**
         * @brief Classify a single document.
         */
        EDiscoveryResult classify(const std::string& doc_id, const std::string& doc_text);

        /**
         * @brief Classify a batch of documents for high throughput.
         */
        std::vector<EDiscoveryResult> classify_batch(const std::map<std::string, std::string>& documents);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::legal