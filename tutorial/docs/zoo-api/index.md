# The `xInfer` Model Zoo API

Welcome to the `xInfer::zoo`. This is the high-level, "batteries-included" API for `xInfer`.

**The `zoo` is a collection of pre-packaged, hyper-optimized, and incredibly easy-to-use solutions for the world's most common AI tasks.**

### The Philosophy: Solutions, Not Just Tools

While the **[Core Toolkit](../core-api/index.md)** provides the powerful, low-level "engine parts" for building custom pipelines, the `zoo` provides the finished "F1 car."

Each class in the `zoo` is a complete, end-to-end pipeline that abstracts away all the complexity of pre-processing, inference, and post-processing. The goal is to let you solve a complex problem like real-time object detection or image generation with just **two lines of C++ code:**

1.  One line to **initialize** the pipeline from a pre-built engine.
2.  One line to **predict**.

This is the power of the `zoo`. It gives you the full, state-of-the-art performance of a custom C++/CUDA/TensorRT application with the simplicity of a high-level library.

---

### Key Features of All `zoo` Classes

- **Performance by Default:** Every `zoo` class is built on top of the hyper-performant `xInfer::core::InferenceEngine` and uses fused CUDA kernels from the `preproc` and `postproc` modules wherever possible.
- **Simple, Task-Oriented API:** You don't interact with raw tensors. You provide a `cv::Mat` image and get back a `std::vector<BoundingBox>` or a `std::string`. The API is designed around the final answer, not the intermediate steps.
- **Seamless Hub Integration:** Most `zoo` classes have special constructors that can download a pre-built, perfectly optimized engine for your hardware directly from the **[Ignition Hub](../hub/index.md)**. This provides a "zero-setup" user experience.
- **Robust and Production-Ready:** These classes are designed to be used directly in your final application. They are efficient, safe, and easy to integrate.

---

## The `zoo` Catalog of Solutions

The `zoo` is organized by domain. Explore the available pipelines below to find the solution you need.

### 🖼️ **Computer Vision**

Tools for understanding and analyzing visual information from images and video. This is the most mature and comprehensive part of the `zoo`.

- **Tasks:** Image Classification, Object Detection, Semantic & Instance Segmentation, Pose Estimation, Face Recognition, OCR, and many more.

➡️ **[Explore the Vision API](./vision.md)**

### ✨ **Generative AI**

Powerful pipelines for creating novel content, from images and audio to 3D models.

- **Tasks:** Text-to-Image (Diffusion), Image Generation (GANs), Super-Resolution, Style Transfer, Text-to-Speech.

➡️ **[Explore the Generative API](./generative.md)**

### 📝 **Natural Language Processing (NLP)**

High-throughput, low-latency solutions for understanding and processing human language.

- **Tasks:** Text Classification, Named Entity Recognition (NER), Sentence Embeddings for RAG, Summarization, and Translation.

➡️ **[Explore the NLP API](./nlp.md)**

### 🎧 **Audio & Signal Processing**

Real-time pipelines for analyzing audio signals, from speech to environmental sounds.

- **Tasks:** Speech Recognition, Audio Classification, Speaker Identification, Music Source Separation.

➡️ **[Explore the Audio & DSP API](./audio.md)**

### 📈 **Time Series**

Specialized solutions for forecasting and analyzing sequential data.

- **Tasks:** Forecasting, Anomaly Detection, and Classification.

➡️ **[Explore the Time Series API](./timeseries.md)**

### 🧊 **3D & Spatial Computing**

Cutting-edge pipelines for processing 3D data from sensors like LIDAR.

- **Tasks:** 3D Reconstruction (Gaussian Splatting), Point Cloud Detection, Point Cloud Segmentation.

➡️ **[Explore the 3D API](./threed.md)**

### 🌍 **Geospatial**

Specialized tools for analyzing satellite and aerial imagery.

- **Tasks:** Building & Road Segmentation, Change Detection, Maritime Object Detection.

➡️ **[Explore the Geospatial API](./geospatial.md)**

### ⚕️ **Medical Imaging**

High-performance pipelines for medical image analysis.

- **Tasks:** Tumor Detection, Cell Segmentation, Retinal Abnormality Scanning, Artery Analysis.

➡️ **[Explore the Medical API](./medical.md)**

### 📄 **Document AI**

Pipelines for understanding the structure and content of documents.

- **Tasks:** Table Extraction, Signature Detection, Handwriting Recognition.

➡️ **[Explore the Document AI API](./document.md)**

### 🚀 **Specialized & RL**

Hyper-specialized, high-value solutions for specific industries and advanced applications.

- **Tasks:** Reinforcement Learning Policies, Financial (HFT) Models, Physics Simulation, Genomics.

➡️ **[Explore the Specialized API](./special.md)**
