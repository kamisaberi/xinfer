#include "opencv_image.h"
#include <opencv2/opencv.hpp>
#include <xinfer/core/logging.h>

namespace xinfer::preproc {

void OpenCVImagePreprocessor::init(const ImagePreprocConfig& config) {
    m_config = config;
}

void OpenCVImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
    if (!src.data) return;

    // 1. Wrap raw data in cv::Mat (Zero Copy if possible)
    int type = (src.format == ImageFormat::GRAY) ? CV_8UC1 : CV_8UC3;
    cv::Mat img(src.height, src.width, type, src.data);

    // 2. Resize
    if (src.width != m_config.target_width || src.height != m_config.target_height) {
        cv::resize(img, img, cv::Size(m_config.target_width, m_config.target_height),
                   0, 0, (int)m_config.resize_mode);
    }

    // 3. Format Conversion (e.g. BGR -> RGB)
    if (src.format == ImageFormat::BGR && m_config.target_format == ImageFormat::RGB) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    // 4. Convert to Float & Normalize
    // img is [H, W, C] uint8. We need float.
    cv::Mat float_img;
    img.convertTo(float_img, CV_32FC3, m_config.norm_params.scale_factor);

    // Subtract Mean / Divide Std
    // OpenCV scalar math is usually faster than manual loops due to internal vectorization
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - m_config.norm_params.mean[i]) / m_config.norm_params.std[i];
    }

    // 5. Write to Tensor (NCHW Layout)
    // This is the heavy part on CPU: rearranging pixels
    float* dst_ptr = static_cast<float*>(dst.data());
    int area = m_config.target_width * m_config.target_height;

    if (m_config.layout_nchw) {
        // Planar: RRR... GGG... BBB...
        for (int i = 0; i < 3; ++i) {
            std::memcpy(dst_ptr + (i * area), channels[i].data, area * sizeof(float));
        }
    } else {
        // Interleaved: RGB RGB RGB (Hardware native usually)
        // We need to merge back if we split them, or just use float_img directly
        cv::merge(channels, float_img);
        std::memcpy(dst_ptr, float_img.data, float_img.total() * float_img.elemSize());
    }
}

} // namespace xinfer::preproc