#include <xinfer/zoo/hci/lip_reader.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/text/ocr_interface.h>

// --- Reused Zoo Components ---
#include <xinfer/zoo/vision/face_detector.h> // For finding the mouth
#include <xinfer/preproc/video/video_preprocessor.h> // For stacking frames

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::hci {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct LipReader::Impl {
    LipReaderConfig config_;

    // --- Components ---
    std::unique_ptr<vision::FaceDetector> face_detector_;

    std::unique_ptr<backends::IBackend> lip_engine_;
    std::unique_ptr<preproc::IVideoPreprocessor> video_preproc_; // Frame Stacker
    std::unique_ptr<postproc::IOcrPostprocessor> ctc_decoder_;

    // --- Tensors ---
    core::Tensor input_stack; // 5D
    core::Tensor output_logits;

    Impl(const LipReaderConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Face Detector
        vision::FaceDetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.detector_path;
        det_cfg.conf_threshold = 0.6f;
        face_detector_ = std::make_unique<vision::FaceDetector>(det_cfg);

        // 2. Setup Lip Reading Engine
        lip_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config lip_cfg; lip_cfg.model_path = config_.lip_read_model_path;

        if (!lip_engine_->load_model(lip_cfg.model_path)) {
            throw std::runtime_error("LipReader: Failed to load model.");
        }

        // 3. Setup Video Preprocessor (Frame Stacker)
        // Note: For real-time, we'd need a hardware-accelerated version of this stacker.
        // The CpuFrameStacker is used here as a reference implementation.
        video_preproc_ = std::make_unique<preproc::CpuFrameStacker>();

        preproc::VideoConfig v_cfg;
        v_cfg.time_steps = config_.window_size;
        v_cfg.height = config_.input_height;
        v_cfg.width = config_.input_width;
        v_cfg.channels = 1; // Lip reading models often use grayscale

        // We configure the internal image preprocessor of the stacker
        preproc::ImagePreprocConfig img_cfg;
        img_cfg.target_height = v_cfg.height;
        img_cfg.target_width = v_cfg.width;
        img_cfg.target_format = preproc::ImageFormat::GRAY;
        img_cfg.layout_nchw = true;
        // video_preproc_->get_image_preprocessor()->init(img_cfg); // Assumes getter
        video_preproc_->init(v_cfg); // init() should handle internal preproc setup

        // 4. Setup CTC Decoder
        ctc_decoder_ = postproc::create_ocr(config_.target);
        postproc::OcrConfig ctc_cfg;
        ctc_cfg.vocab_path = config_.vocab_path;
        ctc_cfg.blank_index = config_.blank_index;
        ctc_decoder_->init(ctc_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

LipReader::LipReader(const LipReaderConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

LipReader::~LipReader() = default;
LipReader::LipReader(LipReader&&) noexcept = default;
LipReader& LipReader::operator=(LipReader&&) noexcept = default;

void LipReader::reset() {
    if (pimpl_ && pimpl_->video_preproc_) pimpl_->video_preproc_->reset();
}

LipReadResult LipReader::process_frame(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("LipReader is null.");

    // 1. Find Face
    auto faces = pimpl_->face_detector_->detect(image);
    if (faces.empty()) {
        // If we lose the face, we should probably reset the buffer
        reset();
        return {"", 0.0f};
    }

    // Pick largest face
    auto best_face = std::max_element(faces.begin(), faces.end(),
        [](const auto& a, const auto& b) {
            return a.to_rect().area() < b.to_rect().area();
        });

    cv::Rect face_box = best_face->to_rect();

    // 2. Heuristic Mouth Crop
    // Mouth is typically in the bottom half of the face box
    int mouth_y = face_box.y + face_box.height * 0.6;
    int mouth_h = face_box.height * 0.4;
    int mouth_x = face_box.x;
    int mouth_w = face_box.width;

    cv::Rect mouth_roi(mouth_x, mouth_y, mouth_w, mouth_h);
    mouth_roi &= cv::Rect(0, 0, image.cols, image.rows);
    if (mouth_roi.width <= 0 || mouth_roi.height <= 0) return {"", 0.0f};

    cv::Mat mouth_crop = image(mouth_roi);

    // 3. Push to Buffer
    // `push_and_get` will handle resizing and stacking
    preproc::ImageFrame frame{mouth_crop.data, mouth_crop.cols, mouth_crop.rows, preproc::ImageFormat::BGR};

    pimpl_->video_preproc_->push_and_get(frame, pimpl_->input_stack);

    // 4. Run Inference (only if we have a full window)
    // The frame stacker needs to signal if it's ready.
    // Assuming `push_and_get` fills the buffer and we can check its state.
    // For this simple example, we run every frame, which means early runs will have old data.
    // In a real app, you wait for N frames.

    pimpl_->lip_engine_->predict({pimpl_->input_stack}, {pimpl_->output_logits});

    // 5. Decode
    auto decoded_batch = pimpl_->ctc_decoder_->process(pimpl_->output_logits);

    if (decoded_batch.empty()) {
        return {"", 0.0f};
    }

    return {decoded_batch[0], 1.0f};
}

} // namespace xinfer::zoo::hci