#include <iostream>
#include <string>
#include <vector>

// --- For the "Easy Button" Zoo API ---
#include <include/zoo/vision/classifier.h>

// --- For the "Power User" Core/Builder API ---
#include <include/builders/onnx_exporter.h>
#include <include/builders/engine_builder.h>
#include <include/core/engine.h>

// --- For generating a dummy trained model from xTorch ---
#include <xtorch/xtorch.h>

/**
 * @brief A helper function to create a dummy trained model and assets.
 *
 * In a real-world scenario, these files would already exist. This function
 * just creates them so the example is fully self-contained and runnable.
 */
void create_dummy_assets() {
    std::cout << "--- Creating dummy assets for demonstration ---\n";

    // 1. Create a dummy xTorch ResNet18 model and save its weights
    xt::models::ResNet18 model(1000); // 1000 classes like ImageNet
    model.to(torch::kCPU);
    model.eval();
    xt::save(model, "dummy_resnet18.weights");
    std::cout << "Saved dummy xTorch model to 'dummy_resnet18.weights'\n";

    // 2. Create a dummy labels file
    std::ofstream labels_file("imagenet_labels.txt");
    for (int i = 0; i < 1000; ++i) {
        labels_file << "class_" << i << "\n";
    }
    labels_file.close();
    std::cout << "Saved dummy labels to 'imagenet_labels.txt'\n";

    // 3. Create a dummy image to run inference on
    cv::Mat dummy_image(224, 224, CV_8UC3, cv::Scalar(100, 50, 200));
    cv::imwrite("input_image.jpg", dummy_image);
    std::cout << "Saved dummy image to 'input_image.jpg'\n";

    std::cout << "---------------------------------------------\n\n";
}

/**
 * @brief Demonstrates the simple, high-level `xinfer::zoo` API.
 *
 * This is the recommended workflow for 99% of users.
 */
void run_zoo_api_example() {
    std::cout << "--- Running Example 1: The 'Easy Button' zoo API ---\n";

    try {
        // --- STEP 1: Build the Optimized Engine (One-time, offline step) ---
        // In a real app, you would run this once to create your .engine file.
        // We use the lower-level builders here to demonstrate the full pipeline.

        // 1a. Create an xTorch model and load the weights
        auto model = std::make_shared<xt::models::ResNet>({2,3,4,6}, 10,3);
        xt::load(model, "dummy_resnet18.weights");

        // 1b. Export the xTorch model to the standard ONNX format
        std::string onnx_path = "resnet18.onnx";
        xinfer::builders::InputSpec input_spec{"input", {1, 3, 224, 224}};
        xinfer::builders::export_to_onnx(*model, {input_spec}, onnx_path);

        // 1c. Build the hyper-optimized TensorRT engine from the ONNX file
        std::string engine_path = "resnet18.engine";
        xinfer::builders::EngineBuilder builder;
        builder.from_onnx(onnx_path)
               .with_fp16() // Enable FP16 for a 2x speedup
               .with_max_batch_size(1);
        builder.build_and_save(engine_path);
        std::cout << "Successfully built TensorRT engine: " << engine_path << "\n\n";

        // --- STEP 2: Use the `zoo::ImageClassifier` for easy inference ---
        // This is what your final application code would look like.

        // 2a. Configure the classifier to use our new engine
        xinfer::zoo::vision::ClassifierConfig config;
        config.engine_path = engine_path;
        config.labels_path = "imagenet_labels.txt";

        // 2b. Create the classifier. Initialization is now instant.
        xinfer::zoo::vision::ImageClassifier classifier(config);
        std::cout << "ImageClassifier initialized successfully.\n";

        // 2c. Load an image and run prediction in a single, clean line of code.
        cv::Mat image = cv::imread("input_image.jpg");
        if (image.empty()) {
            throw std::runtime_error("Failed to load input_image.jpg");
        }

        std::cout << "Running prediction...\n";
        std::vector<xinfer::zoo::vision::ClassificationResult> results = classifier.predict(image, 5);

        // 2d. Print the results.
        std::cout << "\nTop 5 Predictions:\n";
        for (const auto& result : results) {
            std::cout << " - Class: " << result.label
                      << " (ID: " << result.class_id << ")"
                      << ", Confidence: " << result.confidence << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in Zoo API example: " << e.what() << std::endl;
    }
    std::cout << "-----------------------------------------------------\n\n";
}

/**
 * @brief Demonstrates the lower-level Core API for advanced users.
 */
void run_core_api_example() {
    std::cout << "--- Running Example 2: The 'Power User' Core API ---\n";
    try {
        // In this scenario, the user already has a pre-built engine.
        std::string engine_path = "resnet18.engine";

        // 1. Load the engine directly.
        xinfer::core::InferenceEngine engine(engine_path);

        // 2. Manually set up pre- and post-processing.
        auto preprocessor = std::make_unique<xinfer::preproc::ImageProcessor>(224, 224, std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225});

        cv::Mat image = cv::imread("input_image.jpg");

        // Manually create the input tensor
        auto input_shape = engine.get_input_shape(0);
        xinfer::core::Tensor input_tensor(input_shape, xinfer::core::DataType::kFLOAT);

        // Manually run the pre-processor
        preprocessor->process(image, input_tensor);

        // 3. Run inference.
        std::vector<xinfer::core::Tensor> output_tensors = engine.infer({input_tensor});

        // 4. Manually handle the output tensor.
        std::cout << "Core API inference successful.\n";
        std::cout << "Output tensor shape: [ ";
        for (long long dim : output_tensors[0].shape()) {
            std::cout << dim << " ";
        }
        std::cout << "]\n";

    } catch (const std::exception& e) {
        std::cerr << "Error in Core API example: " << e.what() << std::endl;
    }
    std::cout << "----------------------------------------------------\n\n";
}


int main() {
    try {
        create_dummy_assets();
        run_zoo_api_example();
        run_core_api_example();
    } catch (const std::exception& e) {
        std::cerr << "An unhandled exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}