# Hub Usage Guide

The `xInfer` Ignition Hub is designed to be accessed in two primary ways, catering to different levels of control and convenience:

1.  **The Core `hub` API:** For developers who want to manually manage the download and loading process.
2.  **The Integrated `zoo` API:** The recommended, "magical" way to use the hub, where the `zoo` classes handle everything automatically.

---

## Core `hub` API

This is the low-level API for directly interacting with the Ignition Hub. It gives you precise control over the download process.

**Header:** `#include <xinfer/hub/downloader.h>`

### `hub::download_engine`

This is the main function for fetching a pre-built engine file from the cloud.

```cpp
#include <xinfer/hub/downloader.h>

std::string download_engine(
    const std::string& model_id,
    const HardwareTarget& target,
    const std::string& cache_dir = "./xinfer_cache",
    const std::string& hub_url = "https://api.your-ignition-hub.com"
);
```

**Parameters:**
- `model_id` (string): The unique identifier for the model on the hub, e.g., `"yolov8n-coco"`.
- `target` (`HardwareTarget`): A struct specifying the exact hardware and software configuration you need.
- `cache_dir` (string, optional): The local directory where downloaded engines will be stored. `xInfer` will automatically use a cached version if it already exists.
- `hub_url` (string, optional): The base URL of the Ignition Hub API.

**Returns:** A `std::string` containing the local file path to the downloaded (or cached) `.engine` file.

#### **Example: Manually Downloading an Engine**

```cpp
#include <xinfer/hub/downloader.h>
#include <xinfer/core/engine.h>
#include <iostream>

int main() {
    try {
        // 1. Specify the exact engine we need.
        std::string model_id = "resnet50-imagenet";
        xinfer::hub::HardwareTarget my_gpu_target = {
            .gpu_architecture = "RTX_4090",       // Or "sm_89"
            .tensorrt_version = "10.1.0",
            .precision = "FP16"
        };

        // 2. Download the engine file. This will be fast if it's already cached.
        std::cout << "Downloading engine...\n";
        std::string engine_path = xinfer::hub::download_engine(model_id, my_gpu_target);
        
        // 3. Now, use the downloaded engine with the core::InferenceEngine.
        std::cout << "Loading engine: " << engine_path << "\n";
        xinfer::core::InferenceEngine engine(engine_path);
        
        std::cout << "Engine loaded successfully!\n";
        // ... proceed with your custom inference pipeline ...

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

---

## The Integrated `zoo` API (Recommended)

This is the simplest and most powerful way to use the Ignition Hub. The `zoo` classes have special constructors that take a `model_id` and handle the download and initialization process automatically.

This workflow hides all the complexity of downloading, caching, and loading, providing a true one-line solution.

### **Example: Instantiating a Classifier from the Hub**

This example demonstrates how to get a hyper-performant, cloud-optimized `ImageClassifier` running with a single line of code.

```cpp
#include <xinfer/zoo/vision/classifier.h>
#include <xinfer/hub/model_info.h> // The HardwareTarget struct is defined here
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // 1. Define the model we want from the hub and our current hardware.
        std::string model_id = "resnet50-imagenet";
        xinfer::hub::HardwareTarget my_gpu_target;
        my_gpu_target.gpu_architecture = "RTX_4090";
        my_gpu_target.tensorrt_version = "10.1.0";
        my_gpu_target.precision = "FP16";
        
        // 2. Instantiate the classifier with the model_id and target.
        //    This single constructor call does everything:
        //    - Downloads the correct .engine file from the Ignition Hub.
        //    - Downloads the associated labels.txt file.
        //    - Loads the engine and initializes the pre-processor.
        std::cout << "Initializing classifier from Ignition Hub...\n";
        xinfer::zoo::vision::ImageClassifier classifier(model_id, my_gpu_target);
        std::cout << "Classifier ready!\n";

        // 3. The classifier is now ready for immediate, high-performance inference.
        cv::Mat image = cv::imread("my_image.jpg");
        auto results = classifier.predict(image);

        // ... print results ...

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

!!! tip "How it works"
The `zoo` constructors that take a `model_id` are high-level wrappers. Internally, they call `hub::download_engine` and `hub::download_asset` to fetch the necessary files, then they call the regular constructor that takes file paths. This provides the magical, "it just works" user experience.

---

### **Hardware Target Reference**

When specifying your `HardwareTarget`, you need to use the correct string for your GPU architecture. Here are the most common values.

| GPU Architecture Name | **`gpu_architecture` String** | Key GPU Examples |
| :--- | :--- | :--- |
| Hopper | `"H100"` or `"sm_90"` | H100, H200 |
| Ada Lovelace | `"RTX_4090"` or `"sm_89"`| RTX 4090, 4080, RTX 6000 Ada |
| Ampere (Embedded)| `"Jetson_Orin"` or `"sm_87"`| Jetson AGX Orin, Orin Nano |
| Ampere (High-End)| `"RTX_3090"` or `"sm_86"`| RTX 3090, 3080, A100 |
| Turing | `"T4"` or `"sm_75"` | RTX 2080 Ti, T4, Jetson Xavier NX |

You should match the `tensorrt_version` to the version of the TensorRT libraries installed on your system.
