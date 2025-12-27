#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::energy {

    struct GeoFeature {
        std::string type;     // "Fault", "Salt", "Horizon_A"
        float confidence;
        std::vector<cv::Point> contour; // Pixel outline of the feature
    };

    struct InterpretationResult {
        // List of all discrete features found
        std::vector<GeoFeature> features;

        // Color-coded map showing all features
        cv::Mat geo_map;
    };

    struct SeismicConfig {
        // Hardware Target (Seismic volumes are huge, GPU is required)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., fault_segmenter_unet.engine)
        std::string model_path;

        // Input Specs (Seismic data is grayscale)
        int input_width = 256;
        int input_height = 256;

        // Class Mapping (Model dependent)
        // e.g., 0=Background, 1=Fault, 2=Salt, 3=Horizon1
        std::vector<std::string> class_names;
        std::vector<std::vector<uint8_t>> class_colors;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SeismicInterpreter {
    public:
        explicit SeismicInterpreter(const SeismicConfig& config);
        ~SeismicInterpreter();

        // Move semantics
        SeismicInterpreter(SeismicInterpreter&&) noexcept;
        SeismicInterpreter& operator=(SeismicInterpreter&&) noexcept;
        SeismicInterpreter(const SeismicInterpreter&) = delete;
        SeismicInterpreter& operator=(const SeismicInterpreter&) = delete;

        /**
         * @brief Interpret a 2D seismic slice.
         *
         * @param seismic_slice Input image (Grayscale, usually float data).
         * @return Interpretation report.
         */
        InterpretationResult interpret_slice(const cv::Mat& seismic_slice);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::energy