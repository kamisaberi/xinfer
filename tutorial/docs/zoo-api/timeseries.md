# Zoo API: Time-Series

The `xinfer::zoo::timeseries` module provides high-level, optimized pipelines for analyzing and predicting sequential data.

These classes are designed to handle common tasks in domains like finance, IoT (Internet of Things), and predictive maintenance. They abstract away the complexity of sequence model inference, allowing you to get answers from your time-series data with simple, clean C++ code.

---

## `Forecaster`

Performs time-series forecasting. Given a sequence of historical data, it predicts a sequence of future values.

**Header:** `#include <xinfer/zoo/timeseries/forecaster.h>`

```cpp
#include <xinfer/zoo/timeseries/forecaster.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the forecaster.
    //    The engine would be pre-built from a model like N-BEATS or a Transformer.
    xinfer::zoo::timeseries::ForecasterConfig config;
    config.engine_path = "assets/sales_forecaster.engine";
    config.input_sequence_length = 90;  // Model was trained on 90 days of history
    config.output_sequence_length = 14; // Model predicts the next 14 days

    // 2. Initialize the forecaster.
    xinfer::zoo::timeseries::Forecaster forecaster(config);

    // 3. Provide a vector of historical data.
    //    (In a real app, this would come from a database or sensor feed)
    std::vector<float> historical_sales(90); // Must match input_sequence_length
    // ... fill historical_sales with data ...

    // 4. Predict the future values in a single line.
    std::vector<float> forecast = forecaster.predict(historical_sales);

    // 5. Print the results.
    std::cout << "Predicted sales for the next 14 days:\n";
    for (size_t i = 0; i < forecast.size(); ++i) {
        std::cout << "Day " << i + 1 << ": " << forecast[i] << std::endl;
    }
}
```
**Config Struct:** `ForecasterConfig`
**Input:** `std::vector<float>` of historical data.
**Output:** `std::vector<float>` of predicted future values.

---

## `AnomalyDetector`

Analyzes a window of time-series data to detect anomalous patterns or events. This is ideal for predictive maintenance and monitoring.

**Header:** `#include <xinfer/zoo/timeseries/anomaly_detector.h>`

```cpp
#include <xinfer/zoo/timeseries/anomaly_detector.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the anomaly detector.
    //    The engine is typically a reconstruction model like an Autoencoder or LSTM.
    xinfer::zoo::timeseries::AnomalyDetectorConfig config;
    config.engine_path = "assets/machine_vibration_anomaly.engine";
    config.sequence_length = 128; // The model analyzes windows of 128 sensor readings
    config.anomaly_threshold = 0.85f;

    // 2. Initialize.
    xinfer::zoo::timeseries::AnomalyDetector detector(config);

    // 3. Provide a window of recent sensor data.
    std::vector<float> sensor_data_window(128);
    // ... fill sensor_data_window with recent readings ...

    // 4. Predict.
    xinfer::zoo::timeseries::AnomalyResult result = detector.predict(sensor_data_window);

    // 5. Check the result.
    if (result.is_anomaly) {
        std::cout << "ANOMALY DETECTED!\n";
        std::cout << "Anomaly Score: " << result.anomaly_score
                  << " (Threshold: " << config.anomaly_threshold << ")\n";
    } else {
        std::cout << "System normal. Anomaly Score: " << result.anomaly_score << std::endl;
    }
}```
**Config Struct:** `AnomalyDetectorConfig`
**Input:** `std::vector<float>` of sequential data.
**Output Struct:** `AnomalyResult` (contains a boolean flag, the anomaly score, and the reconstruction error per time step).

---

## `Classifier`

Performs classification on a segment of time-series data. This can be used for tasks like identifying the state of a machine or classifying heartbeats from an ECG signal.

**Header:** `#include <xinfer/zoo/timeseries/classifier.h>`

```cpp
#include <xinfer/zoo/timeseries/classifier.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the time-series classifier.
    xinfer::zoo::timeseries::ClassifierConfig config;
    config.engine_path = "assets/ecg_arrhythmia.engine";
    config.labels_path = "assets/ecg_labels.txt"; // e.g., "Normal Beat", "Arrhythmia"
    config.sequence_length = 256; // Model analyzes windows of 256 readings

    // 2. Initialize.
    xinfer::zoo::timeseries::Classifier classifier(config);

    // 3. Provide a segment of time-series data.
    std::vector<float> ecg_segment(256);
    // ... fill ecg_segment with data from an ECG sensor ...

    // 4. Predict.
    xinfer::zoo::timeseries::TimeSeriesClassificationResult result = classifier.predict(ecg_segment);

    // 5. Print the result.
    std::cout << "ECG Segment Classification:\n";
    std::cout << " - Label: " << result.label
              << ", Confidence: " << result.confidence << std::endl;
}
```
**Config Struct:** `ClassifierConfig`
**Input:** `std::vector<float>` of sequential data.
**Output Struct:** `TimeSeriesClassificationResult` (contains class ID, label, and confidence score).
