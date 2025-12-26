#include <xinfer/zoo/medical/artery_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::medical {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ArteryAnalyzer::Impl {
    ArteryConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const ArteryConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ArteryAnalyzer: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::GRAY; // Medical often 1-channel
        pre_cfg.layout_nchw = true;

        // Standard Medical Norm (Windowing usually done before this step)
        // Assuming input is 0-255 uint8 mapped to 0-1 float
        pre_cfg.norm_params.scale_factor = 1.0f / 255.0f;

        preproc_->init(pre_cfg);

        // 3. Setup Segmentation Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        post_cfg.target_width = config_.input_width; // Keep processing at model resolution for speed
        post_cfg.target_height = config_.input_height;
        post_cfg.apply_softmax = false;
        postproc_->init(post_cfg);
    }

    // --- Geometric Analysis ---
    void perform_measurements(const cv::Mat& mask, ArteryResult& result) {
        // 1. Compute Distance Transform
        // Value at pixel = distance to nearest zero pixel (background)
        // For a vessel, the centerline pixel value is the radius.
        cv::Mat dist_map;
        cv::distanceTransform(mask, dist_map, cv::DIST_L2, 5);

        // 2. Skeletonization (Thinning)
        // Extract the 1-pixel wide centerline
        cv::Mat skeleton;
        cv::ximgproc::thinning(mask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
        // Note: If ximgproc not available, use standard Top-Hat morphological skeletonization.
        // Fallback simplified skeleton logic:
        // cv::Mat laplacian;
        // cv::Laplacian(dist_map, laplacian, CV_32F);
        // cv::threshold(laplacian, skeleton, -0.5, 255, cv::THRESH_BINARY_INV); (rough approximation)

        // 3. Trace Branches & Profile Diameter
        std::vector<cv::Point> centerline_pts;
        cv::findNonZero(skeleton, centerline_pts);

        // Organize points into branches (Simplified: treat as one disjoint cloud for profiling)
        // In robust app, use graph traversal (DFS/BFS) on skeleton pixels.

        std::vector<ArteryNode> nodes;
        for (const auto& pt : centerline_pts) {
            float radius_px = dist_map.at<float>(pt);
            float diameter_mm = radius_px * 2.0f * config_.mm_per_pixel;

            nodes.push_back({cv::Point2f((float)pt.x, (float)pt.y), diameter_mm});
        }

        // Store raw nodes as one branch for now
        result.branches.push_back(nodes);

        // 4. Detect Stenosis (Local Minima in Diameter)
        // Logic: Compare current diameter to moving average of neighbors
        if (nodes.size() > 10) {
            int window = 10;
            for (size_t i = window; i < nodes.size() - window; ++i) {
                float current_d = nodes[i].diameter_mm;

                // Calculate healthy context (avg of surroundings)
                float healthy_sum = 0.0f;
                for (int k = 1; k <= window; ++k) {
                    healthy_sum += nodes[i - k].diameter_mm + nodes[i + k].diameter_mm;
                }
                float healthy_d = healthy_sum / (2.0f * window);

                // Check narrowing
                if (healthy_d > 0.1f) { // Avoid div/0 noise
                    float ratio = current_d / healthy_d;
                    float blockage = 1.0f - ratio;

                    if (blockage > config_.stenosis_threshold) {
                        // Check if we already have an event nearby (deduplicate)
                        bool distinct = true;
                        for (const auto& s : result.stenoses) {
                            float dx = s.location.x - nodes[i].position.x;
                            float dy = s.location.y - nodes[i].position.y;
                            if (dx*dx + dy*dy < 100.0f) { // 10px radius
                                distinct = false;
                                break;
                            }
                        }

                        if (distinct) {
                            result.stenoses.push_back({
                                nodes[i].position,
                                blockage,
                                healthy_d,
                                current_d
                            });
                        }
                    }
                }
            }
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

ArteryAnalyzer::ArteryAnalyzer(const ArteryConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ArteryAnalyzer::~ArteryAnalyzer() = default;
ArteryAnalyzer::ArteryAnalyzer(ArteryAnalyzer&&) noexcept = default;
ArteryAnalyzer& ArteryAnalyzer::operator=(ArteryAnalyzer&&) noexcept = default;

ArteryResult ArteryAnalyzer::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ArteryAnalyzer is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    // Handle input channels
    frame.format = (image.channels() == 1) ? preproc::ImageFormat::GRAY : preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference (Segmentation)
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Get Binary Mask)
    // Mask is returned at model resolution
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert Tensor -> cv::Mat
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat low_res_mask(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // Threshold to binary (Class 1 = Vessel)
    cv::Mat binary_mask;
    cv::threshold(low_res_mask, binary_mask, 0, 255, cv::THRESH_BINARY);

    // 4. Resize to Original
    ArteryResult result;
    cv::resize(binary_mask, result.segmentation_mask, image.size(), 0, 0, cv::INTER_NEAREST);

    // 5. Geometric Analysis (Measurements)
    pimpl_->perform_measurements(result.segmentation_mask, result);

    return result;
}

} // namespace xinfer::zoo::medical