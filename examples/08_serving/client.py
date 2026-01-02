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