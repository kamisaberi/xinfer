Here are comprehensive examples for the **Serving Module**.

These examples cover:
1.  **C++ Host:** How to launch the server.
2.  **Python Client:** How to send images/data to the server.
3.  **cURL Client:** Quick testing from the command line.
4.  **Docker Deployment:** How to containerize the service.

### 1. C++ Host Application
**File:** `examples/08_serving/run_server.cpp`

This is the binary that runs on your Edge Device (Jetson, Rockchip, or Server). It starts the HTTP listener and exposes your model folder.

```cpp
#include <xinfer/serving/server.h>
#include <xinfer/core/logging.h>
#include <iostream>
#include <filesystem>

// Simple signal handler for Ctrl+C
#include <csignal>
#include <atomic>

std::atomic<bool> stop_requested(false);

void signal_handler(int) {
    stop_requested = true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./xinfer-server <model_repository_path> [port]" << std::endl;
        return 1;
    }

    std::string repo_path = argv[1];
    int port = (argc > 2) ? std::stoi(argv[2]) : 8080;

    if (!std::filesystem::exists(repo_path)) {
        std::cerr << "Error: Directory does not exist: " << repo_path << std::endl;
        return 1;
    }

    // Configure Server
    xinfer::serving::ServerConfig config;
    config.port = port;
    config.model_repo_path = repo_path;
    config.num_threads = 8; // Adjust based on CPU cores

    xinfer::serving::ModelServer server(config);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << " xInfer Model Server Running" << std::endl;
    std::cout << " Port: " << port << std::endl;
    std::cout << " Repo: " << repo_path << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Run in a separate thread so we can handle exit signals, 
    // or just let it block main thread.
    std::signal(SIGINT, signal_handler);
    
    // Note: server.start() is blocking. 
    // In a real app, you might run it in a std::thread and join it.
    server.start();

    return 0;
}
```

---

### 2. Python Client (End-User)
**File:** `examples/08_serving/client.py`

This script demonstrates how a Data Scientist or Web App would talk to your C++ inference engine. It handles the **image preprocessing** (Resize, Normalize, NCHW Transpose) on the client side before sending the JSON.

```python
import requests
import json
import cv2
import numpy as np
import time

# Configuration
SERVER_URL = "http://localhost:8080"
MODEL_NAME = "yolov8m" # Expects yolov8m.engine or .rknn in the repo
IMAGE_PATH = "street.jpg"

def preprocess_image(image_path, target_size=(640, 640)):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: raise Exception("Image not found")
    
    # 2. Resize
    img = cv2.resize(img, target_size)
    
    # 3. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 4. Normalize (0-255 -> 0.0-1.0)
    img = img.astype(np.float32) / 255.0
    
    # 5. Transpose (HWC -> NCHW)
    # xInfer tensors expect planar format
    img = img.transpose(2, 0, 1) 
    
    # 6. Add Batch Dimension (1, C, H, W)
    img = np.expand_dims(img, axis=0)
    
    # Flatten for JSON serialization
    return img.flatten().tolist(), img.shape

def main():
    # Check Health
    try:
        health = requests.get(f"{SERVER_URL}/health")
        print(f"Server Status: {health.json()}")
    except:
        print("Server is down.")
        return

    # Prepare Data
    print(f"Preprocessing {IMAGE_PATH}...")
    input_data, shape = preprocess_image(IMAGE_PATH)

    payload = {
        "input": input_data,
        "shape": shape
    }

    # Send Request
    print(f"Sending request to model '{MODEL_NAME}'...")
    start = time.time()
    
    response = requests.post(
        f"{SERVER_URL}/v1/models/{MODEL_NAME}:predict",
        json=payload
    )
    
    end = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success!")
        print(f"Inference Time (Server-side): {result['inference_time_ms']} ms")
        print(f"Total Round Trip: {(end-start)*1000:.2f} ms")
        print(f"Output size: {len(result['output'])} floats")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    main()
```

---

### 3. cURL Example (Command Line)
Useful for quick debugging or health checks from a remote terminal.

**Health Check:**
```bash
curl http://localhost:8080/health
# Output: {"status":"ok","version":"xInfer 1.0.0"}
```

**Run Inference (Dummy Data):**
Assuming you have a simple model loaded named `test_model`.
```bash
curl -X POST http://localhost:8080/v1/models/test_model:predict \
     -H "Content-Type: application/json" \
     -d '{
           "shape": [1, 3], 
           "input": [0.5, 0.2, 0.1]
         }'
```

---

### 4. Docker Deployment (`Dockerfile.serving`)

This Dockerfile packages the server for cloud or edge deployment (e.g., Kubernetes or AWS Lambda).

```dockerfile
# Use a base image compatible with your hardware 
# (e.g., nvcr.io/nvidia/l4t-base:r35.1.0 for Jetson, or ubuntu:22.04 for Cloud)
FROM ubuntu:22.04

# 1. Install Dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopencv-dev \
    libcurl4-openssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Source
WORKDIR /app
COPY . .

# 3. Build xInfer Server
RUN mkdir build && cd build && \
    cmake .. -DXINFER_BUILD_SERVING=ON -DXINFER_ENABLE_TRT=OFF -DXINFER_ENABLE_OPENVINO=ON && \
    make -j$(nproc)

# 4. Setup Runtime Env
# Create a folder for models to be mounted later
RUN mkdir /models

# 5. Expose Port
EXPOSE 8080

# 6. Run Server
# Users should mount their model directory to /models
CMD ["./build/examples/xinfer-server", "/models", "8080"]
```

**How to run the Docker container:**
```bash
# Build
docker build -t xinfer-server -f Dockerfile.serving .

# Run (Mount your local model folder to /models inside container)
docker run -p 8080:8080 -v $(pwd)/my_engine_files:/models xinfer-server
```