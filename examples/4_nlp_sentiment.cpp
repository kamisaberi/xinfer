#include <include/zoo/nlp/classifier.h>
#include <iostream>

int main() {
    // Pre-build your BERT engine:
    // xinfer-cli build --onnx sentiment_bert.onnx --save_engine sentiment_bert.engine

    xinfer::zoo::nlp::ClassifierConfig config;
    config.engine_path = "assets/sentiment_bert.engine";
    config.labels_path = "assets/sentiment_labels.txt"; // e.g., "negative", "positive"
    config.vocab_path = "assets/bert_vocab.txt";

    xinfer::zoo::nlp::Classifier classifier(config);

    std::string text = "xInfer is an incredibly fast and easy-to-use library!";
    auto results = classifier.predict(text, 2);

    std::cout << "Sentiment analysis for: \"" << text << "\"\n";
    for (const auto& result : results) {
        printf(" - Label: %-10s, Confidence: %.4f\n", result.label.c_str(), result.confidence);
    }
    return 0;
}