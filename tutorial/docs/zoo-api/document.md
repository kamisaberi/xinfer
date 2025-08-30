# Zoo API: Document AI

The `xinfer::zoo::document` module provides a suite of high-performance pipelines for understanding the structure and content of complex documents like invoices, forms, and reports.

Standard OCR can extract text, but it doesn't understand its meaning or layout. The `zoo` classes in this module are built on top of advanced, multi-modal models that can parse a document's visual and textual elements to extract structured data. These pipelines are hyper-optimized with `xInfer` to enable high-throughput document processing.

---

## `OCR` (Optical Character Recognition)

A complete, two-stage pipeline that first detects the locations of text in an image and then recognizes the characters within each location.

**Header:** `#include <xinfer/zoo/document/ocr.h>`

```cpp
#include <xinfer/zoo/document/ocr.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configure the OCR pipeline with two separate engines.
    xinfer::zoo::document::OCRConfig config;
    config.detection_engine_path = "assets/craft_text_detector.engine";
    config.recognition_engine_path = "assets/crnn_text_recognizer.engine";
    config.character_set = "0123456789abcdefghijklmnopqrstuvwxyz";

    // 2. Initialize.
    xinfer::zoo::document::OCR ocr_pipeline(config);

    // 3. Process a document image.
    cv::Mat image = cv::imread("assets/invoice.png");
    std::vector<xinfer::zoo::document::OCRResult> results = ocr_pipeline.predict(image);

    // 4. Print the extracted text and its location.
    std::cout << "Extracted " << results.size() << " text regions:\n";
    for (const auto& result : results) {
        std::cout << " - Text: \"" << result.text << "\" (Confidence: " << result.confidence << ")\n";
    }
}
```
**Config Struct:** `OCRConfig`
**Input:** `cv::Mat` document image.
**Output Struct:** `OCRResult` (contains the recognized text, its bounding polygon, and a confidence score).

---

## `LayoutParser`

Performs document layout analysis. It segments a document page into its constituent structural elements, such as paragraphs, tables, figures, and headers.

**Header:** `#include <xinfer/zoo/document/layout_parser.h>`
*(Note: The implementation of this class would use a powerful instance segmentation model.)*

```cpp
#include <xinfer/zoo/document/layout_parser.h> // Conceptual header
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::document::LayoutParserConfig config;
    config.engine_path = "assets/layoutlmv3.engine"; // Engine for a model like LayoutLMv3
    config.labels_path = "assets/layout_labels.txt"; // e.g., "paragraph", "table", "figure"

    xinfer::zoo::document::LayoutParser parser(config);

    cv::Mat page_image = cv::imread("assets/report_page.png");
    auto layout_elements = parser.predict(page_image);

    std::cout << "Detected " << layout_elements.size() << " layout elements:\n";
    for (const auto& element : layout_elements) {
        std::cout << " - Found a '" << element.label << "' at bounding box [ "
                  << element.bounding_box.x << ", " << element.bounding_box.y << " ]\n";
    }
}
```
**Config Struct:** `LayoutParserConfig`
**Input:** `cv::Mat` document image.
**Output Struct:** `LayoutElement` (contains the element's bounding box and its class label).

---

## `TableExtractor`

A complex pipeline that detects the structure of a table in an image and uses OCR to extract its contents into a structured, machine-readable format.

**Header:** `#include <xinfer/zoo/document/table_extractor.h>`

```cpp
#include <xinfer/zoo/document/table_extractor.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::document::TableExtractorConfig config;
    config.structure_engine_path = "assets/table_structure_model.engine";
    config.ocr_config_path = "assets/ocr_config.json"; // Path to a config for the internal OCR engine

    xinfer::zoo::document::TableExtractor extractor(config);

    cv::Mat table_image = cv::imread("assets/financial_table.png");
    xinfer::zoo::document::Table extracted_table = extractor.predict(table_image);

    std::cout << "Extracted Table Contents:\n";
    for (const auto& row : extracted_table) {
        for (const auto& cell : row) {
            std::cout << cell << "\t|\t";
        }
        std::cout << "\n";
    }
}
```
**Config Struct:** `TableExtractorConfig`
**Input:** `cv::Mat` image of a table.
**Output:** `Table` (a `std::vector<std::vector<std::string>>`).
**"F1 Car" Technology:** This class is a powerful orchestrator. It internally uses a TensorRT engine for table structure recognition and the `zoo::document::OCR` pipeline for text extraction, combining them with C++ logic to reconstruct the final table.

---

## `SignatureDetector`

Detects the presence and location of handwritten signatures in a document.

**Header:** `#include <xinfer/zoo/document/signature_detector.h>`

```cpp
#include <xinfer/zoo/document/signature_detector.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::document::SignatureDetectorConfig config;
    config.engine_path = "assets/signature_detector.engine";
    
    xinfer::zoo::document::SignatureDetector detector(config);

    cv::Mat contract_page = cv::imread("assets/signed_contract.jpg");
    auto signatures = detector.predict(contract_page);

    std::cout << "Found " << signatures.size() << " signatures.\n";
    // You could then crop these regions for signature verification.
}
```
**Config Struct:** `SignatureDetectorConfig`
**Input:** `cv::Mat` document image.
**Output Struct:** `Signature` (contains bounding box and confidence score).

---

## `HandwritingRecognizer`

A specialized OCR pipeline designed to transcribe handwritten text from an image patch.

**Header:** `#include <xinfer/zoo/document/handwriting_recognizer.h>`

```cpp
#include <xinfer/zoo/document/handwriting_recognizer.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::document::HandwritingRecognizerConfig config;
    config.engine_path = "assets/handwriting_crnn.engine";
    config.character_map_path = "assets/handwriting_chars.txt";

    xinfer::zoo::document::HandwritingRecognizer recognizer(config);

    // Assume we've already detected the location of a handwritten line
    cv::Mat handwritten_line = cv::imread("assets/handwritten_note_line.png");
    auto result = recognizer.predict(handwritten_line);

    std::cout << "Recognized Text: \"" << result.text << "\"\n";
    std::cout << "Confidence: " << result.confidence << "\n";
}
```
**Config Struct:** `HandwritingRecognizerConfig`
**Input:** `cv::Mat` image of a single line of handwritten text.
**Output Struct:** `HandwritingRecognitionResult` (contains the transcribed text and a confidence score).