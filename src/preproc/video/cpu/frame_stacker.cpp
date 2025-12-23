#include <xinfer/preproc/video/video_preprocessor.h>
#include <xinfer/preproc/image/cpu/opencv_image.h> // Reuse single frame preproc logic
#include <deque>
#include <cstring>

namespace xinfer::preproc {

class CpuFrameStacker : public IVideoPreprocessor {
public:
    void init(const VideoConfig& config) override {
        m_config = config;

        // Setup internal image preprocessor configuration
        ImagePreprocConfig img_cfg;
        img_cfg.target_width = config.width;
        img_cfg.target_height = config.height;
        img_cfg.layout_nchw = true; // We usually want Channels first per frame
        m_image_preproc.init(img_cfg);

        m_frame_size_bytes = config.width * config.height * config.channels * sizeof(float);
        reset();
    }

    void reset() override {
        // Clear buffer and fill with zeros (or duplicate first frame logic)
        m_buffer.clear();

        // Create blank frame
        core::Tensor blank_frame;
        blank_frame.resize({(int64_t)m_config.channels, (int64_t)m_config.height, (int64_t)m_config.width}, core::DataType::kFLOAT);
        std::memset(blank_frame.data(), 0, blank_frame.size() * sizeof(float));

        // Pre-fill buffer to avoid cold start errors
        for(int i=0; i<m_config.time_steps; ++i) {
            m_buffer.push_back(blank_frame);
        }
    }

    void push_and_get(const ImageFrame& new_frame, core::Tensor& dst_stack) override {
        // 1. Process the single new frame (Resize/Norm)
        core::Tensor processed_frame;
        // Allocate space for 1 frame
        processed_frame.resize({(int64_t)m_config.channels, (int64_t)m_config.height, (int64_t)m_config.width}, core::DataType::kFLOAT);
        m_image_preproc.process(new_frame, processed_frame);

        // 2. Update Ring Buffer
        m_buffer.pop_front(); // Remove oldest
        m_buffer.push_back(processed_frame); // Add newest

        // 3. Stack into Output Tensor
        // Output Shape: [1, T, C, H, W]
        dst_stack.resize({1, (int64_t)m_config.time_steps, (int64_t)m_config.channels, (int64_t)m_config.height, (int64_t)m_config.width}, core::DataType::kFLOAT);

        char* dst_ptr = static_cast<char*>(dst_stack.data());

        // Perform contiguous copy
        // Optimization: In a GPU version, this would be a circular buffer pointer math
        // rather than memcpys, but for CPU this is standard.
        for (const auto& frame : m_buffer) {
            std::memcpy(dst_ptr, frame.data(), m_frame_size_bytes);
            dst_ptr += m_frame_size_bytes;
        }
    }

private:
    VideoConfig m_config;
    std::deque<core::Tensor> m_buffer; // The Ring Buffer
    OpenCVImagePreprocessor m_image_preproc;
    size_t m_frame_size_bytes = 0;
};

}