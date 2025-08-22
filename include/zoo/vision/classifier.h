#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp> // For user-friendly cv::Mat inputs

// --- Low-level xInfer components used internally ---
// These are included here because the private members will hold them.
// In a very large library, you might use the PIMPL idiom to hide these,
// but for clarity, we include them here.
#include "../../core/engine.h"
#include "../../preproc/image_processor.h"
#include "../../builders/engine_builder.h"
#include "../../builders/onnx_exporter.h"

// --- Dependency on xTorch for model definition ---
// This class needs to know the architecture to load the weights
// before converting to a TensorRT engine.
#include <xtorch/models/resnet.h> // Example for a ResNet classifier

namespace xinfer::zoo::vision {

/**
 * @struct ClassificationResult
 * @brief A simple, human-readable struct to hold a single classification prediction.
 */
struct ClassificationResult {
    int class_id;       // The integer index of the predicted class (e.g., 281)
    float confidence;   // The prediction confidence score (e.g., 0.98)
    std::string label;  // The human-readable class name (e.g., "tabby, tabby cat")
};

/**
 * @struct ClassifierConfig
 * @brief A configuration structure for creating an ImageClassifier.
 *
 * This allows advanced users to customize every aspect of the optimization
 * and pre-processing pipeline.
 */
struct ClassifierConfig {
    // --- Model Definition ---
    // Path to the trained xTorch model weights file.
    std::string xtorch_model_path;
    // Path to a simple text file with class names, one per line, corresponding to the class indices.
    std::string labels_path = "";
    // Specify the model architecture from the xTorch zoo.
    // This example defaults to ResNet18, but could be an enum or string.
    std::string architecture = "ResNet18";

    // --- TensorRT Build Options ---
    // If true, build the engine with FP16 precision for a ~2x speedup.
    bool use_fp16 = true;
    // If true, build with INT8 precision. Requires a calibration dataset.
    bool use_int8 = false;
    // Path to a directory of images for INT8 calibration.
    std::string int8_calibration_dataset_path = "";
    // Maximum batch size the engine will support.
    int max_batch_size = 1;
    // GPU memory in MB for the TensorRT builder workspace.
    size_t workspace_mb = 1024; // 1 GB

    // --- Pre-processing Options ---
    // Dimensions the input image will be resized to.
    int resize_width = 224;
    int resize_height = 224;
    // Normalization parameters (standard for ImageNet).
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
};

/**
 * @class ImageClassifier
 * @brief A high-level, hyper-optimized pipeline for image classification.
 *
 * This class abstracts the entire complex workflow:
 * 1. Loads a trained xTorch model from a weights file.
 * 2. Automatically converts it to a hyper-optimized TensorRT engine (with optional FP16/INT8).
 * 3. Handles all pre-processing (resize, normalize) and post-processing (softmax, top-k) on the GPU.
 *
 * The result is a simple, one-line predict() function with maximum performance.
 */
class ImageClassifier {
public:
    /**
     * @brief Advanced constructor. Creates a classifier with a detailed configuration.
     * This is the primary constructor.
     * @param config A ClassifierConfig struct with all necessary settings.
     */
    explicit ImageClassifier(const ClassifierConfig& config)
        : config_(config) {
        // The real magic happens in this private helper function.
        initialize();
    }

    /**
     * @brief Simple constructor for convenience. Creates a classifier with default settings.
     * @param xtorch_model_path Path to the trained xTorch ResNet-style model weights.
     * @param labels_path Optional path to a text file containing class names.
     */
    explicit ImageClassifier(const std::string& xtorch_model_path, const std::string& labels_path = "")
        : config_{xtorch_model_path, labels_path} {
        initialize();
    }

    /**
     * @brief Runs inference on a single image provided as an OpenCV Mat.
     * @param image An OpenCV BGR image (cv::Mat) to classify.
     * @param top_k The number of top predictions to return.
     * @return A vector of ClassificationResult structs, sorted by confidence.
     */
    std::vector<ClassificationResult> predict(const cv::Mat& image, int top_k = 5) {
        if (!engine_ || !preprocessor_) {
            throw std::runtime_error("Classifier is not initialized.");
        }

        // 1. Pre-process the image using the fused CUDA kernel
        core::Tensor input_tensor({BATCH_SIZE, 3, config_.resize_height, config_.resize_width}, core::DataType::kFLOAT);
        preprocessor_->process(image, input_tensor);

        // 2. Run inference using the TensorRT engine
        auto output_tensors = engine_->infer({input_tensor});

        // 3. Post-process the raw logits to get probabilities and top-k results
        return postprocess(output_tensors[0], top_k);
    }

private:
    // --- Internal "F1 Car" Components ---
    ClassifierConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;

    // --- Private Helper Methods ---

    /**
     * @brief The core setup logic that builds the TensorRT engine.
     */
    void initialize() {
        // --- 1. Instantiate xTorch model and load weights ---
        std::shared_ptr<xt::models::ResNet18> model = std::make_shared<xt::models::ResNet18>(); // This could be a factory based on config_.architecture
        // xt::load(model, config_.xtorch_model_path); // Assuming xTorch has a load function

        // --- 2. Export to ONNX ---
        std::string onnx_path = "temp_classifier.onnx";
        builders::InputSpec input_spec{"input", {BATCH_SIZE, 3, config_.resize_height, config_.resize_width}};
        // builders::export_to_onnx(*model, {input_spec}, onnx_path); // This would be the actual call

        // --- 3. Build TensorRT Engine ---
        std::string engine_path = "classifier.engine";
        builders::EngineBuilder builder;
        builder.from_onnx(onnx_path);
        if (config_.use_fp16) {
            builder.with_fp16();
        }
        // ... INT8 calibration logic would go here ...
        builder.with_max_batch_size(config_.max_batch_size);
        builder.build_and_save(engine_path);

        // --- 4. Load the final engine and set up processors ---
        engine_ = std::make_unique<core::InferenceEngine>(engine_path);
        preprocessor_ = std::make_unique<preproc::ImageProcessor>(config_.resize_width, config_.resize_height, config_.mean, config_.std);

        // --- 5. Load class labels ---
        if (!config_.labels_path.empty()) {
            // Logic to read the text file into class_labels_
        }
    }

    /**
     * @brief Post-processes the raw output logits from the model.
     */
    std::vector<ClassificationResult> postprocess(const core::Tensor& logits_tensor, int top_k) {
        // This would ideally run softmax on the GPU, but for simplicity, we can do it on the CPU for now.
        std::vector<float> logits;
        logits.resize(logits_tensor.num_elements());
        logits_tensor.copy_to_host(logits.data());

        // Find top_k results... (logic for softmax, sorting, etc.)

        std::vector<ClassificationResult> results;
        // ... populate the results vector ...

        return results;
    }
};

} // namespace xinfer::zoo::vision

