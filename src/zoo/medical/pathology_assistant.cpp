#include <xinfer/zoo/medical/pathology_assistant.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <numeric>
#include <random>

namespace xinfer::zoo::medical {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PathologyAssistant::Impl {
    PathologyConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Visualization LUT
    std::vector<cv::Vec3b> color_lut_;

    Impl(const PathologyConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("PathologyAssistant: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB; // H&E is RGB
        pre_cfg.layout_nchw = true;

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = 1.0f; // Input 0-255

        preproc_->init(pre_cfg);

        // 3. Setup Segmentation Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        post_cfg.target_width = config_.input_width; // Analyze at model res
        post_cfg.target_height = config_.input_height;
        postproc_->init(post_cfg);

        // 4. Setup Colors
        init_colors();
    }

    void init_colors() {
        if (!config_.class_colors.empty()) {
            for (const auto& c : config_.class_colors) {
                if (c.size() >= 3) color_lut_.push_back(cv::Vec3b(c[2], c[1], c[0])); // RGB->BGR
            }
        } else {
            // Defaults
            // 0: Black (Background), 1: Green (Normal), 2: Red (Tumor), 3: Yellow (Stroma)
            color_lut_ = {
                cv::Vec3b(0, 0, 0),       // Background
                cv::Vec3b(0, 255, 0),     // Normal
                cv::Vec3b(0, 0, 255),     // Tumor
                cv::Vec3b(0, 255, 255)    // Stroma
            };
        }
    }

    // --- Statistics: Calculate Tumor Burden ---
    void compute_stats(const cv::Mat& mask, PathologyResult& result) {
        int total_pixels = mask.rows * mask.cols;

        // Count pixels per class
        // Typically Class 0 is "Glass" (Empty slide area), we exclude it from total tissue.
        std::map<int, int> counts;
        int tissue_pixels = 0;

        if (mask.isContinuous()) {
            const uint8_t* ptr = mask.ptr<uint8_t>(0);
            for (int i = 0; i < total_pixels; ++i) {
                int id = ptr[i];
                counts[id]++;
                if (id != 0) tissue_pixels++; // Assuming 0 is Background
            }
        } else {
            // Slower iterator
            for(int y=0; y<mask.rows; ++y) {
                for(int x=0; x<mask.cols; ++x) {
                    int id = mask.at<uint8_t>(y,x);
                    counts[id]++;
                    if (id != 0) tissue_pixels++;
                }
            }
        }

        // Calculate Percentages relative to Tissue Area
        float tumor_pixels = 0;

        for (const auto& kv : counts) {
            int id = kv.first;
            int count = kv.second;

            std::string name = (id < (int)config_.class_names.size()) ? config_.class_names[id] : std::to_string(id);

            if (tissue_pixels > 0 && id != 0) {
                result.tissue_percentages[name] = (float)count / tissue_pixels;
            } else {
                result.tissue_percentages[name] = 0.0f;
            }

            // Check if this class is Tumor (Naive check by name substring or ID logic)
            // Assuming config provided proper names or ID 2 is tumor based on default LUT
            if (name.find("Tumor") != std::string::npos || id == 2) {
                tumor_pixels += count;
            }
        }

        if (tissue_pixels > 0) {
            result.tumor_burden = tumor_pixels / tissue_pixels;
        } else {
            result.tumor_burden = 0.0f;
        }

        result.has_malignancy = (result.tumor_burden > config_.tumor_threshold);
    }

    cv::Mat create_overlay(const cv::Mat& mask) {
        cv::Mat color_mask(mask.size(), CV_8UC3);

        int rows = mask.rows;
        int cols = mask.cols;
        if (mask.isContinuous() && color_mask.isContinuous()) {
            cols *= rows;
            rows = 1;
        }

        const uint8_t* p_idx = mask.ptr<uint8_t>(0);
        cv::Vec3b* p_dst = color_mask.ptr<cv::Vec3b>(0);
        size_t lut_size = color_lut_.size();

        for (int i = 0; i < cols * rows; ++i) {
            uint8_t id = p_idx[i];
            if (id < lut_size) {
                p_dst[i] = color_lut_[id];
            } else {
                p_dst[i] = cv::Vec3b(128, 128, 128); // Unknown
            }
        }
        return color_mask;
    }
};

// =================================================================================
// Public API
// =================================================================================

PathologyAssistant::PathologyAssistant(const PathologyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PathologyAssistant::~PathologyAssistant() = default;
PathologyAssistant::PathologyAssistant(PathologyAssistant&&) noexcept = default;
PathologyAssistant& PathologyAssistant::operator=(PathologyAssistant&&) noexcept = default;

PathologyResult PathologyAssistant::analyze_patch(const cv::Mat& patch) {
    if (!pimpl_) throw std::runtime_error("PathologyAssistant is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = patch.data;
    frame.width = patch.cols;
    frame.height = patch.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert to Mat
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat low_res_mask(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // Resize to original patch size (NN)
    cv::Mat final_mask;
    if (patch.size() != low_res_mask.size()) {
        cv::resize(low_res_mask, final_mask, patch.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        final_mask = low_res_mask.clone();
    }

    // 4. Analysis
    PathologyResult result;
    result.tissue_mask = final_mask;

    // Calculate Stats
    pimpl_->compute_stats(final_mask, result);

    // Create visualization (Color Mask)
    // Note: User can blend this with original image
    cv::Mat color_mask = pimpl_->create_overlay(final_mask);

    // Blend for convenience (30% transparency)
    cv::addWeighted(patch, 0.7, color_mask, 0.3, 0.0, result.tissue_mask);

    return result;
}

} // namespace xinfer::zoo::medical