#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::timeseries {

    struct TimeSeriesClassificationResult {
        int class_id;
        float confidence;
        std::string label;
    };

    struct ClassifierConfig {
        std::string engine_path;
        std::string labels_path = "";
        int sequence_length = 128;
    };

    class Classifier {
    public:
        explicit Classifier(const ClassifierConfig& config);
        ~Classifier();

        Classifier(const Classifier&) = delete;
        Classifier& operator=(const Classifier&) = delete;
        Classifier(Classifier&&) noexcept;
        Classifier& operator=(Classifier&&) noexcept;

        TimeSeriesClassificationResult predict(const std::vector<float>& time_series_window);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::timeseries

