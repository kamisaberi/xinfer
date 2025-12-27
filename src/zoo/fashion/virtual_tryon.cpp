#include <xinfer/zoo/fashion/virtual_tryon.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- We reuse other Zoo modules to build this complex pipeline ---
#include <xinfer/zoo/vision/pose_estimator.h>
#include <xinfer/zoo/vision/segmenter.h>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>

#include <iostream>
#include <vector>

namespace xinfer::zoo::fashion {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct VirtualTryOn::Impl {
    TryOnConfig config_;

    // --- Components ---
    // Stage 1
    std::unique_ptr<vision::PoseEstimator> pose_estimator_;
    std::unique_ptr<vision::Segmenter> person_segmenter_;

    // Stage 2
    std::unique_ptr<backends::IBackend> warp_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> warp_preproc_;

    // Stage 3
    std::unique_ptr<backends::IBackend> gen_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> gen_preproc_;

    // --- Tensors ---
    core::Tensor pose_kpts_tensor, person_mask_tensor;
    core::Tensor cloth_img_tensor, warped_cloth_tensor;
    core::Tensor gen_input_tensor, final_image_tensor;

    Impl(const TryOnConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Init Pose Estimator
        vision::PoseEstimatorConfig pose_cfg;
        pose_cfg.target = config_.target;
        pose_cfg.model_path = config_.pose_model_path;
        pose_estimator_ = std::make_unique<vision::PoseEstimator>(pose_cfg);

        // 2. Init Person Segmenter
        vision::SegmenterConfig seg_cfg;
        seg_cfg.target = config_.target;
        seg_cfg.model_path = config_.seg_model_path;
        seg_cfg.input_width = 256; // Smaller res for fast seg
        seg_cfg.input_height = 256;
        person_segmenter_ = std::make_unique<vision::Segmenter>(seg_cfg);

        // 3. Init Warping Model
        warp_engine_ = backends::BackendFactory::create(config_.target);
        if (!warp_engine_->load_model(config_.warp_model_path))
            throw std::runtime_error("Failed to load Warp model.");

        warp_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig wp_cfg;
        wp_cfg.target_width = config_.input_width;
        wp_cfg.target_height = config_.input_height;
        warp_preproc_->init(wp_cfg);

        // 4. Init Generator
        gen_engine_ = backends::BackendFactory::create(config_.target);
        if (!gen_engine_->load_model(config_.generator_model_path))
            throw std::runtime_error("Failed to load Generator model.");

        gen_preproc_ = preproc::create_image_preprocessor(config_.target);
        gen_preproc_->init(wp_cfg); // Same config
    }

    // Custom post-processing to convert tensor to image
    cv::Mat tensor_to_image(const core::Tensor& t) {
        // [1,3,H,W] float [-1,1] -> BGR uint8
        auto shape = t.shape();
        int h = shape[2], w = shape[3];
        const float* ptr = (const float*)t.data();
        cv::Mat img(h, w, CV_8UC3);
        // ... conversion logic ...
        return img;
    }
};

// =================================================================================
// Public API
// =================================================================================

VirtualTryOn::VirtualTryOn(const TryOnConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

VirtualTryOn::~VirtualTryOn() = default;
VirtualTryOn::VirtualTryOn(VirtualTryOn&&) noexcept = default;
VirtualTryOn& VirtualTryOn::operator=(VirtualTryOn&&) noexcept = default;

cv::Mat VirtualTryOn::try_on(const cv::Mat& person_image, const cv::Mat& clothing_image) {
    if (!pimpl_) throw std::runtime_error("VirtualTryOn is null.");

    // --- 1. Get Person Representation ---
    // A. Pose
    auto poses = pimpl_->pose_estimator_->estimate(person_image);
    if (poses.empty()) return person_image; // No person found

    // Convert keypoints to a flat tensor
    pimpl_->pose_kpts_tensor.resize({1, 1, 18, 2}, core::DataType::kFLOAT); // Shape [1,1,18,2] for OpenPose
    // ... fill tensor ...

    // B. Segmentation Mask
    auto seg_res = pimpl_->person_segmenter_->segment(person_image);
    // Convert cv::Mat mask to a float tensor
    // ... fill tensor (pimpl_->person_mask_tensor) ...

    // C. Preprocess clothing image
    preproc::ImageFrame cloth_frame{clothing_image.data, clothing_image.cols, clothing_image.rows, preproc::ImageFormat::BGR};
    pimpl_->warp_preproc_->process(cloth_frame, pimpl_->cloth_img_tensor);

    // --- 2. Geometric Warping ---
    // The warper takes [Clothing, PoseKeypoints] and outputs a warped clothing image
    pimpl_->warp_engine_->predict({pimpl_->cloth_img_tensor, pimpl_->pose_kpts_tensor}, {pimpl_->warped_cloth_tensor});

    // --- 3. Final Synthesis (GAN) ---
    // The generator takes a concatenation of inputs:
    // [PersonImage_masked, PoseKeypoints, PersonMask, WarpedCloth]
    // We prepare a multi-channel tensor here.

    // (Preprocessing for this step is complex and omitted for brevity)

    pimpl_->gen_engine_->predict({/*multi-channel input tensor*/}, {pimpl_->final_image_tensor});

    // --- 4. Post-process ---
    return pimpl_->tensor_to_image(pimpl_->final_image_tensor);
}

} // namespace xinfer::zoo::fashion