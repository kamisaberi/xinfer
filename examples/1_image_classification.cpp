#include <include/zoo/vision/classifier.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // In a real app, the .engine file would be pre-built with the xinfer-cli
    // xinfer-cli build --onnx resnet18.onnx --save_engine resnet18.engine --fp16

    // 1. Configure the classifier
    xinfer::zoo::vision::ClassifierConfig config;
    config.engine_path = "assets/resnet18.engine"; // Path to your pre-built engine
    config.labels_path = "assets/imagenet_labels.txt";

    // 2. Initialize the classifier. This is a fast, one-time setup.
    xinfer::zoo::vision::ImageClassifier classifier(config);

    // 3. Load an image and predict in a single line
    cv::Mat image = cv::imread("assets/cat_image.jpg");
    std::vector<xinfer::zoo::vision::ClassificationResult> results = classifier.predict(image, 5);

    // 4. Print the results
    std::cout << "Top 5 Predictions for cat_image.jpg:\n";
    for (const auto& result : results) {
        printf(" - Class: %-25s (ID: %d), Confidence: %.4f\n",
               result.label.c_str(), result.class_id, result.confidence);
    }
    return 0;
}