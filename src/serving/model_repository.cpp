#include <xinfer/serving/model_repository.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace xinfer::serving {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ModelRepository::Impl {
    std::string repo_path_;

    // Map: Model Name -> Backend Instance
    // This owns the memory of the loaded models.
    std::map<std::string, std::unique_ptr<backends::IBackend>> loaded_models_;

    // Map: Model Name -> File Path (for lazy loading)
    std::map<std::string, std::string> available_files_;

    // Mutex for thread-safe access map modification
    mutable std::mutex mutex_;

    Impl(const std::string& path) : repo_path_(path) {
        scan_directory();
    }

    void scan_directory() {
        if (!fs::exists(repo_path_)) {
            XINFER_LOG_ERROR("Model repository path does not exist: " + repo_path_);
            return;
        }

        XINFER_LOG_INFO("Scanning model repository: " + repo_path_);

        // Prioritized extensions list
        // If "model.onnx" and "model.engine" exist, we want .engine to take precedence.
        // We handle this by scanning logic or simple overwriting.

        for (const auto& entry : fs::directory_iterator(repo_path_)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                std::string name_no_ext = entry.path().stem().string();

                // Check supported extensions
                if (is_supported_extension(ext)) {
                    // Store available file
                    // Logic: If duplicates exist (e.g. yolo.onnx and yolo.engine),
                    // we prefer hardware specific formats over generic ONNX.

                    bool register_file = true;
                    if (available_files_.count(name_no_ext)) {
                        std::string existing_ext = fs::path(available_files_[name_no_ext]).extension().string();
                        if (get_priority(existing_ext) > get_priority(ext)) {
                            register_file = false;
                        }
                    }

                    if (register_file) {
                        available_files_[name_no_ext] = entry.path().string();
                    }
                }
            }
        }
    }

    // Helper: Is this a file xInfer can load?
    bool is_supported_extension(const std::string& ext) {
        static const std::vector<std::string> exts = {
            ".engine", ".plan",  // NVIDIA
            ".rknn",             // Rockchip
            ".xmodel",           // AMD/Xilinx
            ".xml",              // OpenVINO
            ".bin",              // QNN/Microchip
            ".hef",              // Hailo
            ".tflite",           // Edge TPU / Mobile
            ".onnx"              // Generic
        };
        return std::find(exts.begin(), exts.end(), ext) != exts.end();
    }

    // Helper: Priority for conflicting filenames
    int get_priority(const std::string& ext) {
        if (ext == ".engine" || ext == ".rknn" || ext == ".xmodel" || ext == ".hef") return 10;
        if (ext == ".xml" || ext == ".bin" || ext == ".tflite") return 5;
        if (ext == ".onnx") return 1;
        return 0;
    }

    // Helper: Map extension to Target enum
    Target determine_target(const std::string& path) {
        std::string ext = fs::path(path).extension().string();

        if (ext == ".engine" || ext == ".plan") return Target::NVIDIA_TRT;
        if (ext == ".rknn")                     return Target::ROCKCHIP_RKNN;
        if (ext == ".xmodel")                   return Target::AMD_VITIS;
        if (ext == ".xml")                      return Target::INTEL_OV;
        if (ext == ".hef")                      return Target::HAILO_RT;
        if (ext == ".tflite") {
            // Could be Google TPU or Generic CPU.
            // Check for "_edgetpu" suffix convention
            if (path.find("_edgetpu") != std::string::npos) return Target::GOOGLE_TPU;
            return Target::INTEL_OV; // Or generic TFLite runner
        }
        if (ext == ".bin")                      return Target::QUALCOMM_QNN; // Or Intel FPGA/Microchip, ambiguous

        // Default fallback for ONNX
        return Target::INTEL_OV;
    }

    backends::IBackend* load(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. Already loaded?
        if (loaded_models_.count(name)) {
            return loaded_models_[name].get();
        }

        // 2. Check availability
        if (available_files_.find(name) == available_files_.end()) {
            return nullptr;
        }

        std::string path = available_files_[name];
        Target target = determine_target(path);

        XINFER_LOG_INFO("ModelRepository: Loading '" + name + "' for target " + std::to_string((int)target));

        // 3. Create Backend
        auto backend = backends::BackendFactory::create(target);
        if (!backend) {
            XINFER_LOG_ERROR("Failed to create backend factory for target ID: " + std::to_string((int)target));
            return nullptr;
        }

        // 4. Load Model File
        // Note: For some backends (like QNN/Microchip), we might need extra config.
        // For the server, we assume standard defaults.
        if (!backend->load_model(path)) {
            XINFER_LOG_ERROR("Backend failed to load model file: " + path);
            return nullptr;
        }

        // 5. Store in Cache
        loaded_models_[name] = std::move(backend);
        return loaded_models_[name].get();
    }
};

// =================================================================================
// Public API
// =================================================================================

ModelRepository::ModelRepository(const std::string& repo_path)
    : pimpl_(std::make_unique<Impl>(repo_path)) {}

ModelRepository::~ModelRepository() = default;
ModelRepository::ModelRepository(ModelRepository&&) noexcept = default;
ModelRepository& ModelRepository::operator=(ModelRepository&&) noexcept = default;

backends::IBackend* ModelRepository::get_model(const std::string& model_name) {
    if (!pimpl_) return nullptr;
    return pimpl_->load(model_name);
}

bool ModelRepository::exists(const std::string& model_name) const {
    if (!pimpl_) return false;
    std::lock_guard<std::mutex> lock(pimpl_->mutex_);
    return pimpl_->available_files_.count(model_name) > 0;
}

void ModelRepository::unload_model(const std::string& model_name) {
    if (!pimpl_) return;
    std::lock_guard<std::mutex> lock(pimpl_->mutex_);
    if (pimpl_->loaded_models_.count(model_name)) {
        XINFER_LOG_INFO("Unloading model: " + model_name);
        pimpl_->loaded_models_.erase(model_name);
    }
}

std::vector<std::string> ModelRepository::list_models() const {
    if (!pimpl_) return {};
    std::lock_guard<std::mutex> lock(pimpl_->mutex_);

    std::vector<std::string> names;
    for (const auto& pair : pimpl_->available_files_) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace xinfer::serving