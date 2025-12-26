#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::timeseries {

    /**
     * @brief Result of Time Series Classification.
     */
    struct TSClassResult {
        int id;             // Class Index
        float confidence;   // Probability (0.0 - 1.0)
        std::string label;  // Class Name (e.g., "Walking", "Fall_Detected")
    };

    struct TSClassifierConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., har_cnn.onnx, gesture_lstm.engine)
        std::string model_path;

        // Path to labels file (newline separated)
        std::string labels_path;

        // Window Size (Sequence Length)
        // The model expects inputs of shape [1, window_size, features]
        int window_size = 50;

        // Number of features per time step (e.g. 3 for Accel X/Y/Z)
        int num_features = 3;

        // Normalization (Standard Scaling: (x - mean) / std)
        std::vector<float> mean;
        std::vector<float> std;

        // Input Layout:
        // true:  [Batch, Time, Features] (Standard RNN/Transformer)
        // false: [Batch, Features, Time] (Standard 1D-CNN)
        bool layout_time_first = true;

        // Post-processing
        int top_k = 1;
    };

    class Classifier {
    public:
        explicit Classifier(const TSClassifierConfig& config);
        ~Classifier();

        // Move semantics
        Classifier(Classifier&&) noexcept;
        Classifier& operator=(Classifier&&) noexcept;
        Classifier(const Classifier&) = delete;
        Classifier& operator=(const Classifier&) = delete;

        /**
         * @brief Push a new data point into the sliding window.
         *
         * @param features Vector of size 'num_features'.
         * @return True if window is full and ready for classification.
         */
        bool push(const std::vector<float>& features);

        /**
         * @brief Classify the current window.
         *
         * @return Vector of top_k results.
         */
        std::vector<TSClassResult> classify();

        /**
         * @brief Clear the internal sliding window history.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::timeseries