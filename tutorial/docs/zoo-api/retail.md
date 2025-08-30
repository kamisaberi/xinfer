# Zoo API: Retail

The `xinfer::zoo::retail` module provides a suite of high-performance pipelines specifically designed for the unique challenges of the retail industry.

In retail, efficiency is everything. From optimizing the supply chain to understanding customer behavior in real-time, the ability to process vast amounts of data quickly and cost-effectively is a critical competitive advantage. The `zoo` classes in this module are end-to-end solutions that leverage `xInfer`'s hyper-optimized engines to automate core retail operations.

---

## `ShelfAuditor`

Provides a complete solution for automated shelf monitoring. It uses a high-performance object detection model to audit product availability and placement on store shelves.

**Header:** `#include <xinfer/zoo/retail/shelf_auditor.h>`

### Use Case: Real-Time Out-of-Stock Detection

A store manager or a robot can capture an image of an aisle. The `ShelfAuditor` instantly processes this image to identify which products are missing or misplaced, creating an actionable alert for store associates.

```cpp
#include <xinfer/zoo/retail/shelf_auditor.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configure the auditor.
    //    The engine would be a YOLO model trained on specific product SKUs.
    xinfer::zoo::retail::ShelfAuditorConfig config;
    config.engine_path = "assets/product_detector.engine";
    config.labels_path = "assets/product_skus.txt";
    config.confidence_threshold = 0.7f;

    // 2. Initialize.
    xinfer::zoo::retail::ShelfAuditor auditor(config);

    // 3. Process an image of a store shelf.
    cv::Mat shelf_image = cv::imread("assets/aisle_4_snapshot.jpg");
    std::vector<xinfer::zoo::retail::ShelfItem> results = auditor.audit(shelf_image);

    // 4. Print the inventory count.
    std::cout << "Shelf Audit Results:\n";
    for (const auto& item : results) {
        std::cout << " - Item: " << item.label
                  << " (ID: " << item.class_id << ")"
                  << ", Count: " << item.count << "\n";
    }
    // This data can be compared against the store's inventory database to find discrepancies.
}
```
**Config Struct:** `ShelfAuditorConfig`
**Input:** `cv::Mat` image of a store shelf.
**Output Struct:** `ShelfItem` (contains product ID, label, on-shelf count, and locations).
**"F1 Car" Technology:** This class is a specialized wrapper around the `zoo::vision::ObjectDetector`, using its high-throughput capabilities to count hundreds of items in a single frame.

---

## `CustomerAnalyzer`

Performs real-time, anonymous tracking of customers in a physical store to generate analytics on traffic patterns and behavior.

**Header:** `#include <xinfer/zoo/retail/customer_analyzer.h>`

### Use Case: Store Layout Optimization

A store manager wants to understand how customers move through the store to optimize product placement and identify bottlenecks. The `CustomerAnalyzer` processes video feeds to generate a continuous stream of tracking data and a long-term traffic heatmap.

```cpp
#include <xinfer/zoo/retail/customer_analyzer.h>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Configure the analyzer.
    //    This uses two engines: one for person detection and one for pose estimation.
    xinfer::zoo::retail::CustomerAnalyzerConfig config;
    config.detection_engine_path = "assets/person_detector.engine";
    config.pose_engine_path = "assets/pose_estimator.engine";

    // 2. Initialize.
    xinfer::zoo::retail::CustomerAnalyzer analyzer(config);

    // 3. Process frames from a store's security camera in a loop.
    cv::VideoCapture cap("assets/store_footage.mp4");
    cv::Mat frame;
    while (cap.read(frame)) {
        // This call updates the internal state of all tracked customers.
        std::vector<xinfer::zoo::retail::TrackedCustomer> tracked_customers = analyzer.track(frame);
        
        // (In a real app, you would use this tracking data for further analysis)
    }

    // 4. After processing, generate a long-term traffic heatmap.
    cv::Mat heatmap = analyzer.generate_heatmap();
    cv::imwrite("traffic_heatmap.png", heatmap);
    std::cout << "Saved customer traffic heatmap to traffic_heatmap.png\n";
}
```
**Config Struct:** `CustomerAnalyzerConfig`
**Input:** `cv::Mat` video frame.
**Output Struct:** `TrackedCustomer` (contains a unique track ID and bounding box).
**Method:** `generate_heatmap()` returns a `cv::Mat` visualization.

---

## `DemandForecaster`

A specialized version of the time-series forecaster, tailored for predicting retail product demand.

**Header:** `#include <xinfer/zoo/retail/demand_forecaster.h>`

### Use Case: Automated Inventory Replenishment

A retail chain needs to automatically forecast the demand for thousands of products to optimize their inventory and prevent "out of stock" situations.

```cpp
#include <xinfer/zoo/retail/demand_forecaster.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the forecaster.
    xinfer::zoo::retail::DemandForecasterConfig config;
    config.engine_path = "assets/sales_forecasting_model.engine";
    config.input_sequence_length = 90;  // Use 90 days of sales history
    config.output_sequence_length = 14; // Predict demand for the next 14 days

    // 2. Initialize.
    xinfer::zoo::retail::DemandForecaster forecaster(config);

    // 3. Provide the historical sales data for a single product.
    std::vector<float> historical_sales(90);
    // ... load the last 90 days of sales data from a database ...

    // 4. Predict the future demand.
    std::vector<float> predicted_demand = forecaster.predict(historical_sales);

    std::cout << "Predicted demand for the next 14 days:\n";
    for (size_t i = 0; i < predicted_demand.size(); ++i) {
        std::cout << " - Day " << i+1 << ": " << (int)predicted_demand[i] << " units\n";
    }
}
```
**Config Struct:** `DemandForecasterConfig`
**Input:** `std::vector<float>` of historical sales data.
**Output:** `std::vector<float>` of predicted future sales.
**"F1 Car" Technology:** This class is a specialized wrapper around the `zoo::timeseries::Forecaster`, which can use a hyper-optimized Mamba or Transformer engine to capture complex seasonalities and trends in the sales data.