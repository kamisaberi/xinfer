#include <iostream>
#include <vector>
#include <string>

#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    // Target: Rockchip RK3588 (Edge SIEM Collector)
    Target target = Target::ROCKCHIP_RKNN;

    // 1. Define Schema (Matches training data)
    std::vector<preproc::tabular::ColumnSchema> schema = {
        {"timestamp", preproc::tabular::ColumnType::TIMESTAMP, preproc::tabular::EncodingType::MIN_MAX_SCALE},
        {"src_ip",    preproc::tabular::ColumnType::IP_ADDRESS, preproc::tabular::EncodingType::IP_SPLIT}, // 4 floats
        {"dst_ip",    preproc::tabular::ColumnType::IP_ADDRESS, preproc::tabular::EncodingType::IP_SPLIT}, // 4 floats
        {"protocol",  preproc::tabular::ColumnType::CATEGORICAL, preproc::tabular::EncodingType::LABEL_ENCODE,
                      .category_map = {{"TCP", 0.0f}, {"UDP", 1.0f}, {"ICMP", 2.0f}}},
        {"bytes",     preproc::tabular::ColumnType::NUMERICAL, preproc::tabular::EncodingType::STANDARD_SCALE,
                      .mean = 500.0f, .std = 120.0f}
    };

    // 2. Setup Components
    auto encoder = preproc::create_tabular_preprocessor(target);
    encoder->init(schema);

    auto engine = backends::BackendFactory::create(target);
    engine->load_model("autoencoder_siem.rknn");

    // Anomaly Detector (Calculates reconstruction error)
    auto anomaly_proc = postproc::create_anomaly(target);
    postproc::AnomalyConfig anom_cfg;
    anom_cfg.threshold = 0.05f; // MSE Threshold
    anomaly_proc->init(anom_cfg);

    // 3. Process a "Live" Log Line
    // "167888.0", "192.168.1.50", "10.0.0.5", "TCP", "99999" (Huge byte count!)
    preproc::tabular::TableRow log_row = {"167888.0", "192.168.1.50", "10.0.0.5", "TCP", "99999"};

    core::Tensor input, output;

    // Encode: IP "192.168.1.50" -> [0.75, 0.66, 0.004, 0.19]
    encoder->process(log_row, input);

    // Inference (Autoencoder tries to reconstruct valid traffic)
    engine->predict({input}, {output});

    // Check for Anomaly
    // If output is very different from input, it's an anomaly
    auto result = anomaly_proc->process(input, output);

    if (result.is_anomaly) {
        std::cout << "[ALERT] Anomaly Detected! Score: " << result.anomaly_score << std::endl;
        std::cout << "        Suspicious Log: Src=" << log_row[1] << " Bytes=" << log_row[4] << std::endl;
    } else {
        std::cout << "[OK] Traffic Normal." << std::endl;
    }

    return 0;
}