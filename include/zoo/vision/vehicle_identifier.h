#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct VehicleAttributes {
        std::string type;
        float type_confidence;
        std::string color;
        float color_confidence;
        std::string make;
        float make_confidence;
        std::string model;
        float model_confidence;
    };

    struct VehicleIdentifierConfig {
        std::string engine_path;
        std::string type_labels_path = "";
        std::string color_labels_path = "";
        std::string make_labels_path = "";
        std::string model_labels_path = "";
        int input_width = 224;
        int input_height = 224;
    };

    class VehicleIdentifier {
    public:
        explicit VehicleIdentifier(const VehicleIdentifierConfig& config);
        ~VehicleIdentifier();

        VehicleIdentifier(const VehicleIdentifier&) = delete;
        VehicleIdentifier& operator=(const VehicleIdentifier&) = delete;
        VehicleIdentifier(VehicleIdentifier&&) noexcept;
        VehicleIdentifier& operator=(VehicleIdentifier&&) noexcept;

        VehicleAttributes predict(const cv::Mat& vehicle_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

