# Zoo API: Natural Language Processing (NLP)

The `xinfer::zoo::nlp` module provides a suite of high-performance pipelines for common Natural Language Processing tasks.

These classes are built on top of hyper-optimized TensorRT engines for state-of-the-art Transformer and sequence models. They are designed to bring high-throughput, low-latency language understanding to your native C++ applications, enabling tasks from real-time sentiment analysis to complex document processing.

A core component for these pipelines is a **tokenizer**, which converts raw text into integer IDs that the models can understand. `xInfer` assumes you will manage tokenization using a library like SentencePiece or Hugging Face Tokenizers C++, and the `zoo` classes will take these token IDs as input.

---

## `Classifier`

Performs text classification. Given a piece of text, it assigns it to a pre-defined category (e.g., sentiment, topic, intent).

**Header:** `#include <xinfer/zoo/nlp/classifier.h>`

```cpp
#include <xinfer/zoo/nlp/classifier.h>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // 1. Configure the text classifier.
    //    The engine would be a pre-built BERT or DistilBERT model.
    xinfer::zoo::nlp::ClassifierConfig config;
    config.engine_path = "assets/sentiment_bert.engine";
    config.labels_path = "assets/sentiment_labels.txt"; // e.g., "negative", "positive"
    config.vocab_path = "assets/bert_vocab.txt"; // For the internal tokenizer

    // 2. Initialize.
    xinfer::zoo::nlp::Classifier classifier(config);

    // 3. Predict the sentiment of a sentence.
    std::string text = "xInfer is an incredibly fast and easy-to-use library!";
    auto results = classifier.predict(text, 2); // Get top 2 results

    // 4. Print the results.
    std::cout << "Sentiment analysis for: \"" << text << "\"\n";
    for (const auto& result : results) {
        printf(" - Label: %-10s, Confidence: %.4f\n", result.label.c_str(), result.confidence);
    }
}```
**Config Struct:** `ClassifierConfig`
**Input:** `std::string` of text.
**Output Struct:** `TextClassificationResult` (contains class ID, label, and confidence).

---

## `Embedder`

Converts a piece of text into a fixed-size, high-dimensional vector (an "embedding") that captures its semantic meaning. This is the backbone of modern semantic search and RAG systems.

**Header:** `#include <xinfer/zoo/nlp/embedder.h>`

```cpp
#include <xinfer/zoo/nlp/embedder.h>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // 1. Configure the embedder.
    //    The engine would be a pre-built Sentence-BERT model.
    xinfer::zoo::nlp::EmbedderConfig config;
    config.engine_path = "assets/sentence_bert.engine";
    config.vocab_path = "assets/bert_vocab.txt";

    // 2. Initialize.
    xinfer::zoo::nlp::Embedder embedder(config);

    // 3. Create embeddings for a list of sentences.
    std::vector<std::string> texts = {
        "The cat sat on the mat.",
        "A feline was resting on the rug."
    };
    std::vector<xinfer::zoo::nlp::TextEmbedding> embeddings = embedder.predict_batch(texts);

    // 4. Compare the embeddings using cosine similarity.
    float similarity = xinfer::zoo::nlp::Embedder::compare(embeddings, embeddings);

    std::cout << "Semantic similarity between the two sentences: " << similarity << std::endl;
}
```
**Config Struct:** `EmbedderConfig`
**Input:** `std::string` or `std::vector<std::string>`.
**Output:** `TextEmbedding` (a `std::vector<float>`).

---

## `NER` (Named Entity Recognition)

Scans a piece of text and extracts named entities like people, organizations, and locations.

**Header:** `#include <xinfer/zoo/nlp/ner.h>`

```cpp
#include <xinfer/zoo/nlp/ner.h>
#include <iostream>
#include <string>

int main() {
    // 1. Configure the NER pipeline.
    xinfer::zoo::nlp::NERConfig config;
    config.engine_path = "assets/ner_bert.engine";
    config.labels_path = "assets/ner_labels.txt"; // e.g., "B-PER", "I-PER", "B-ORG"
    config.vocab_path = "assets/bert_vocab.txt";

    // 2. Initialize.
    xinfer::zoo::nlp::NER ner_pipeline(config);

    // 3. Predict.
    std::string text = "Apple Inc. was founded by Steve Jobs in Cupertino.";
    auto entities = ner_pipeline.predict(text);

    // 4. Print the extracted entities.
    std::cout << "Found " << entities.size() << " entities:\n";
    for (const auto& entity : entities) {
        std::cout << " - Text: \"" << entity.text << "\", Label: " << entity.label << "\n";
    }
}
```
**Config Struct:** `NERConfig`
**Input:** `std::string`.
**Output Struct:** `NamedEntity` (contains the text, label, score, and position).

---

## `QuestionAnswering`

Finds the answer to a question within a given context paragraph.

**Header:** `#include <xinfer/zoo/nlp/question_answering.h>`

```cpp
#include <xinfer/zoo/nlp/question_answering.h>
#include <iostream>
#include <string>

int main() {
    xinfer::zoo::nlp::QAConfig config;
    config.engine_path = "assets/qa_bert.engine";
    config.vocab_path = "assets/bert_vocab.txt";

    xinfer::zoo::nlp::QuestionAnswering qa_pipeline(config);

    std::string context = "xInfer is a C++ library designed for high-performance inference. It uses NVIDIA TensorRT to optimize models.";
    std::string question = "What technology does xInfer use?";

    auto result = qa_pipeline.predict(question, context);

    std::cout << "Question: " << question << "\n";
    std::cout << "Answer: " << result.answer << " (Score: " << result.score << ")\n";
}
```
**Config Struct:** `QAConfig`
**Input:** `std::string` for question and `std::string` for context.
**Output Struct:** `QAResult` (contains the answer text, score, and position).

---

## `TextGenerator` / `CodeGenerator`

Provides an interface for running generative Large Language Models (LLMs) for text or code completion.

**Header:** `#include <xinfer/zoo/nlp/text_generator.h>`

```cpp
#include <xinfer/zoo/nlp/text_generator.h>
#include <iostream>
#include <string>

int main() {
    xinfer::zoo::nlp::TextGeneratorConfig config;
    config.engine_path = "assets/llama3_8b.engine";
    config.vocab_path = "assets/llama_vocab.json";
    config.max_new_tokens = 100;

    xinfer::zoo::nlp::TextGenerator generator(config);

    std::string prompt = "xInfer is a C++ library that enables ";
    
    std::cout << "Prompt: " << prompt;
    // The streaming function calls the lambda for each new piece of text generated.
    generator.predict_stream(prompt, [](const std::string& token_str) {
        std::cout << token_str << std::flush;
    });
    std::cout << std::endl;
}
```
**Config Struct:** `TextGeneratorConfig` / `CodeGeneratorConfig`
**Methods:** `.predict()` for a single string, and `.predict_stream()` for real-time, token-by-token streaming.
