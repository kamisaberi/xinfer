#include <xinfer/serving/model_repository.h>
#include <xinfer/core/logging.h>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

namespace xinfer::serving {

struct ModelRepository::Impl {
    std::string repo_path;

    // Cache: Name -> Backend Instance
    std::map<std::string, std::unique_ptr<backends::IBackend>> loaded_models_;
    std::mutex mutex_;

    Impl(const std::string& path) : repo_path(path) {}

    // Logic to map file extension to xInfer Target
    Target detect_target(const std::string& ext) {
        if (ext == ".engine" || ext == ".plan") return Target::NVIDIA_TRT;
        if (ext == ".rknn") return Target::ROCKCHIP_RKNN;
        if (ext == ".xml") return Target::INTEL_OV;
        if (ext == ".xmodel") return Target::AMD_VITIS;
        if (ext == ".hef") return Target::HAILO_RT;
        if (ext == ".bin") return Target::QUALCOMM_QNN; // Ambiguous, but common default
        if (ext == ".tflite") return Target::GOOGLE_TPU;
        return Target::INTEL_OV; // Default fallback
    }

    backends::IBackend* load(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. Check Cache
        if (loaded_models_.count(name)) {
            return loaded_models_[name].get();
        }

        // 2. Scan for file
        // We look for any file with the model name and a supported extension
        std::string found_path = "";
        Target target = Target::INTEL_OV;

        if (!fs::exists(repo_path)) {
            XINFER_LOG_ERROR("Repository path not found: " + repo_path);
            return nullptr;
        }

        for (const auto& entry : fs::directory_iterator(repo_path)) {
            if (entry.path().stem() == name) {
                found_path = entry.path().string();
                target = detect_target(entry.path().extension().string());
                break;
            }
        }

        if (found_path.empty()) {
            XINFER_LOG_ERROR("Model not found in repo: " + name);
            return nullptr;
        }

        // 3. Create Backend
        XINFER_LOG_INFO("Loading model: " + name + " (Target: " + std::to_string((int)target) + ")");
        auto backend = backends::BackendFactory::create(target);

        if (!backend->load_model(found_path)) {
            XINFER_LOG_ERROR("Failed to load backend for: " + name);
            return nullptr;
        }

        loaded_models_[name] = std::move(backend);
        return loaded_models_[name].get();
    }
};

ModelRepository::ModelRepository(const std::string& repo_path)
    : pimpl_(std::make_unique<Impl>(repo_path)) {}

ModelRepository::~ModelRepository() = default;

backends::IBackend* ModelRepository::get_model(const std::string& model_name) {
    return pimpl_->load(model_name);
}

void ModelRepository::unload_model(const std::string& model_name) {
    std::lock_guard<std::mutex> lock(pimpl_->mutex_);
    pimpl_->loaded_models_.erase(model_name);
}

}