#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

// xInfer Headers
#include <xinfer/preproc/factory.h>
#include <xinfer/core/tensor.h>
#include <xinfer/core/logging.h>

using namespace xinfer;
using namespace xinfer::preproc;
using namespace xinfer::preproc::tabular;

// Helper to print a tensor slice
void print_tensor_features(const core::Tensor& t, int count) {
    const float* data = static_cast<const float*>(t.data());
    std::cout << "[ ";
    for (int i = 0; i < count; ++i) {
        std::cout << std::fixed << std::setprecision(3) << data[i] << " ";
    }
    std::cout << "... ] (Total: " << t.size() << ")" << std::endl;
}

int main() {
    // -------------------------------------------------------------------------
    // 1. SETUP
    // -------------------------------------------------------------------------
    // We use CPU implementation (LogEncoder) via the factory.
    // Even if Target is ROCKCHIP, the factory returns the optimized CPU class
    // because tabular processing (parsing) is faster on CPU than moving small data to NPU.
    auto preprocessor = create_tabular_preprocessor(Target::INTEL_OV);

    std::cout << "--- xInfer SIEM Log Parser Demo ---" << std::endl;

    // Define the Schema (usually loaded from a JSON config in production)
    std::vector<ColumnSchema> schema;

    // Col 0: Timestamp (Standard Scaled)
    schema.push_back({"timestamp", ColumnType::TIMESTAMP, EncodingType::STANDARD_SCALE,
                      .mean = 1670000000.0f, .std = 500000.0f});

    // Col 1: Source IP (Split into 4 normalized floats)
    schema.push_back({"src_ip", ColumnType::IP_ADDRESS, EncodingType::IP_SPLIT});

    // Col 2: Destination IP
    schema.push_back({"dst_ip", ColumnType::IP_ADDRESS, EncodingType::IP_SPLIT});

    // Col 3: Protocol (Categorical -> Label Encoding)
    schema.push_back({
        "protocol", ColumnType::CATEGORICAL, EncodingType::LABEL_ENCODE,
        .category_map = {{"TCP", 0.0f}, {"UDP", 1.0f}, {"ICMP", 2.0f}, {"HTTP", 3.0f}},
        .unknown_value = -1.0f
    });

    // Col 4: Bytes Sent (Numerical -> MinMax Scaled)
    schema.push_back({"bytes", ColumnType::NUMERICAL, EncodingType::MIN_MAX_SCALE,
                      .min = 0.0f, .max = 65535.0f});

    // Col 5: Log Message (Ignored by the Anomaly Detector, handled by NLP module separately)
    schema.push_back({"msg", ColumnType::IGNORE, EncodingType::NONE});

    // Initialize
    preprocessor->init(schema);
    std::cout << "Schema Initialized. Output Feature Width: " << preprocessor->get_output_width() << std::endl;
    // Expected Width:
    // Time(1) + SrcIP(4) + DstIP(4) + Proto(1) + Bytes(1) = 11 floats

    // -------------------------------------------------------------------------
    // 2. SINGLE ROW DEMO
    // -------------------------------------------------------------------------
    std::cout << "\n[Test Case 1] Normal Traffic" << std::endl;

    // Raw log: TS, Src, Dst, Proto, Bytes, Msg
    TableRow row = {"1670000500", "192.168.1.10", "8.8.8.8", "UDP", "128", "DNS Query"};

    core::Tensor output_tensor;
    preprocessor->process(row, output_tensor);

    print_tensor_features(output_tensor, 11);

    // Verify IP Parsing logic (192.168.1.10)
    // 192/255 = 0.753, 168/255 = 0.659, 1/255 = 0.004, 10/255 = 0.039
    const float* ptr = (float*)output_tensor.data();
    std::cout << "  > Source IP (192.168.1.10) Encoded: "
              << ptr[1] << ", " << ptr[2] << ", " << ptr[3] << ", " << ptr[4] << std::endl;

    // -------------------------------------------------------------------------
    // 3. BENCHMARK (High-Throughput Ingestion)
    // -------------------------------------------------------------------------
    std::cout << "\n[Benchmark] Processing 1 Million Logs..." << std::endl;

    // Pre-allocate a large batch of identical rows to simulate load
    int batch_size = 10000;
    int iterations = 100; // Total 1M
    std::vector<TableRow> batch(batch_size, row);
    core::Tensor batch_output;

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<iterations; ++i) {
        // process_batch optimizes memory access patterns
        preprocessor->process_batch(batch, batch_output);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double total_events = batch_size * iterations;
    double eps = total_events / (elapsed_ms / 1000.0);

    std::cout << "  > Processed " << total_events << " events in " << elapsed_ms << " ms." << std::endl;
    std::cout << "  > Throughput: " << (int)eps << " EPS (Events Per Second)" << std::endl;

    if (eps > 100000) {
        std::cout << "  > RESULT: PASSED High-Performance Requirement for Blackbox SIEM." << std::endl;
    } else {
        std::cout << "  > RESULT: WARNING - Optimization needed." << std::endl;
    }

    return 0;
}