#include <include/utils/image_utils.h>
#include <stdexcept>
#include <vector>

namespace xinfer::utils {

cv::Mat tensor_to_mat(const core::Tensor& tensor, const DenormalizationParams& params) {
    if (tensor.dtype() != xinfer::core::DataType::kFLOAT) {
        throw std::invalid_argument("tensor_to_mat currently only supports float tensors.");
    }

    const auto& shape = tensor.shape();
    if (shape.size() != 4 && shape.size() != 3) {
        throw std::invalid_argument("Input tensor must have 3 or 4 dimensions (C, H, W or B, C, H, W).");
    }

    const int C = (shape.size() == 4) ? shape[1] : shape[0];
    const int H = (shape.size() == 4) ? shape[2] : shape[1];
    const int W = (shape.size() == 4) ? shape[3] : shape[2];

    if (C != 1 && C != 3) {
        throw std::invalid_argument("Input tensor must have 1 or 3 channels.");
    }
    if (C == 3 && (params.mean.size() != 3 || params.std.size() != 3)) {
        throw std::invalid_argument("DenormalizationParams must have 3 values for a 3-channel image.");
    }

    // 1. Copy the entire tensor data from GPU to a single CPU buffer
    std::vector<float> h_data(tensor.num_elements());
    tensor.copy_to_host(h_data.data());

    // 2. Create the final destination cv::Mat (HWC layout)
    cv::Mat final_image(H, W, (C == 3) ? CV_8UC3 : CV_8UC1);

    // 3. Loop through the image pixels and perform denormalization and layout conversion (CHW -> HWC)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (C == 3) {
                // Get values from the planar CHW buffer
                float r = h_data[0 * H * W + y * W + x];
                float g = h_data[1 * H * W + y * W + x];
                float b = h_data[2 * H * W + y * W + x];

                // Denormalize
                r = (r * params.std[0] + params.mean[0]) * 255.0f;
                g = (g * params.std[1] + params.mean[1]) * 255.0f;
                b = (b * params.std[2] + params.mean[2]) * 255.0f;

                // Clamp to [0, 255] and write to the interleaved HWC cv::Mat
                // OpenCV expects BGR order
                final_image.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>(b);
                final_image.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(g);
                final_image.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(r);
            } else { // C == 1
                float val = h_data[y * W + x];
                val = (val * params.std[0] + params.mean[0]) * 255.0f;
                final_image.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
            }
        }
    }

    return final_image;
}


bool save_tensor_as_image(const core::Tensor& tensor, const std::string& filepath, const DenormalizationParams& params) {
    try {
        cv::Mat image = tensor_to_mat(tensor, params);
        return cv::imwrite(filepath, image);
    } catch (const std::exception& e) {
        std::cerr << "Error saving tensor as image: " << e.what() << std::endl;
        return false;
    }
}

} // namespace xinfer::utils