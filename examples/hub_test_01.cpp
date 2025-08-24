#include <include/zoo/vision/classifier.h>
#include <include/hub/model_info.h> // For the HardwareTarget struct
#include <opencv2/opencv.hpp>

int main() {
    // 1. Specify the model and the exact hardware target
    xinfer::hub::HardwareTarget my_gpu = {"RTX_4090", "10.1.0", "FP16"};

    // 2. Instantiate the classifier. It will download the correct engine from the cloud.
    xinfer::zoo::vision::ImageClassifier classifier("resnet18-imagenet", my_gpu);

    // 3. Run inference.
    cv::Mat image = cv::imread("my_image.jpg");
    auto results = classifier.predict(image);

    // ...
    return 0;
}