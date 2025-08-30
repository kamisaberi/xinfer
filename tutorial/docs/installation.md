# Installation

Welcome to `xInfer`! This guide will walk you through the process of setting up your environment and building the library from source.

## Prerequisites

`xInfer` is a high-performance library that sits on top of the NVIDIA software stack. Before you begin, you must have the following components installed on your system.

### 1. NVIDIA Driver

You need a recent NVIDIA driver that supports your GPU and CUDA Toolkit version. You can check your driver version by running:
```bash
nvidia-smi
```

### 2. NVIDIA CUDA Toolkit

`xInfer` is built on CUDA. We recommend **CUDA 11.8** or newer.
- **Verification:** Check if `nvcc` is installed and in your PATH:
  ```bash
  nvcc --version
  ```
- **Installation:** If you don't have it, download and install it from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

### 3. NVIDIA cuDNN

TensorRT requires the cuDNN library for accelerating deep learning primitives. We recommend **cuDNN 8.6** or newer.
- **Verification:** Check for the cuDNN header file:
  ```bash
  ls /usr/include/cudnn.h
  ```
- **Installation:** Download and install it from the [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive). Make sure to follow the installation instructions for copying the header and library files to your CUDA Toolkit directory.

### 4. NVIDIA TensorRT

TensorRT is the core optimization engine used by `xInfer`. We recommend **TensorRT 8.6** or newer.
- **Verification:** Check if the TensorRT header `NvInfer.h` exists:
  ```bash
  ls /usr/include/x86_64-linux-gnu/NvInfer.h
  ```
  (The path may vary depending on your installation method).
- **Installation:** Download and install it from the [NVIDIA TensorRT page](https://developer.nvidia.com/tensorrt). The `.deb` or `.rpm` package installation is the easiest method.

### 5. `xTorch` Library

The `xInfer::builders` and `zoo` modules are designed to work seamlessly with models trained or defined in `xTorch`.
- **Installation:** You must build and install `xTorch` first. Please follow the instructions at the **[xTorch GitHub Repository](https://github.com/your-username/xtorch)**.
  ```bash
  git clone https://github.com/your-username/xtorch.git
  cd xtorch
  mkdir build && cd build
  cmake ..
  make -j
  sudo make install
  ```

### 6. Other Dependencies (Compiler, CMake, OpenCV)

You will need a modern C++ compiler and the build tools.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev libcurl4-openssl-dev
```
- **CMake:** Version 3.18 or newer is required.
- **OpenCV:** Required for image processing in the `zoo` and examples.

---

## Building `xInfer` from Source (Recommended)

This is the standard method for building and installing `xInfer`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/xinfer.git
cd xinfer
```

### Step 2: Configure with CMake

Create a build directory and run CMake. This command will find all the dependencies you installed and prepare the build files.

```bash
mkdir build
cd build
cmake ..
```

!!! tip "Troubleshooting Dependency Issues"
If CMake has trouble finding a specific library (especially TensorRT if you installed it from a `.zip` file), you can give it a hint with the `-D` flag. For example:
```bash
cmake -DTensorRT_ROOT=/path/to/your/TensorRT-10.x.x.x ..
```

### Step 3: Compile

Use `make` to compile the library. This will create the `libxinfer.so` shared library and all the example executables.

```bash
# Use -j to specify the number of parallel jobs to speed up compilation
make -j$(nproc)
```

### Step 4: (Optional) Run Tests

After a successful build, you can run the example executable to verify that everything is working.

```bash
# From the build directory
./xinfer_example```
You should see the output from the example programs, indicating a successful build.

### Step 5: (Optional) Install System-Wide

If you want to use `xInfer` as a dependency in other C++ projects on your system, you can install it. This will copy the library files and headers to system directories (like `/usr/local/lib` and `/usr/local/include`).

```bash
sudo make install
```
After this step, other CMake projects can find and use your library with a simple `find_package(xinfer REQUIRED)` command.

---

## Next Steps

Congratulations! You have successfully built and installed `xInfer`.

- **Check out the [🚀 Quickstart](./quickstart.md)** guide to run your first high-performance inference pipeline.
- **Explore the [Model Zoo API](./zoo-api/index.md)** to see all the pre-packaged solutions you can use out-of-the-box.
