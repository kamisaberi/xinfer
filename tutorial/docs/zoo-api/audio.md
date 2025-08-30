# Zoo API: Audio & Speech

The `xinfer::zoo::audio` module provides a suite of high-performance pipelines for the most common audio and speech processing tasks.

These classes are built on top of `xInfer`'s custom CUDA pre-processing kernels and hyper-optimized TensorRT engines. They are designed for latency-critical applications like real-time transcription, voice recognition, and audio analysis, where the performance of C++ is a fundamental requirement.

All pipelines in this module are powered by the `xinfer::preproc::AudioProcessor`, which performs the entire waveform-to-spectrogram conversion on the GPU for maximum efficiency.

---

## `Classifier`

Classifies a short audio clip into a predefined set of categories.

**Header:** `#include <xinfer/zoo/audio/classifier.h>`

### Use Case: Environmental Sound Detection

An application needs to identify sounds in its environment, such as a "siren," "dog bark," or "glass breaking," for security or contextual awareness.

```cpp
#include <xinfer/zoo/audio/classifier.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the audio classifier.
    xinfer::zoo::audio::ClassifierConfig config;
    config.engine_path = "assets/esc50_classifier.engine";
    config.labels_path = "assets/esc50_labels.txt";
    // The audio_config should match the model's training parameters.
    config.audio_config.sample_rate = 16000;

    // 2. Initialize.
    xinfer::zoo::audio::Classifier classifier(config);

    // 3. Load a raw audio waveform (e.g., from a .wav file).
    std::vector<float> waveform; // Assume this is loaded with a 2-second audio clip
    
    // 4. Predict the top 3 sound classes.
    auto results = classifier.predict(waveform, 3);

    // 5. Print the results.
    std::cout << "Top 3 Audio Classifications:\n";
    for (const auto& result : results) {
        printf(" - Label: %-20s, Confidence: %.4f\n", result.label.c_str(), result.confidence);
    }
}
```
**Config Struct:** `ClassifierConfig`
**Input:** `std::vector<float>` audio waveform.
**Output Struct:** `AudioClassificationResult`.

---

## `SpeechRecognizer`

Transcribes spoken language from an audio waveform into text.

**Header:** `#include <xinfer/zoo/audio/speech_recognizer.h>`

```cpp
#include <xinfer/zoo/audio/speech_recognizer.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the speech recognizer.
    xinfer::zoo::audio::SpeechRecognizerConfig config;
    config.engine_path = "assets/whisper_base_en.engine";
    config.character_map_path = "assets/whisper_chars.txt";
    config.audio_config.sample_rate = 16000;

    // 2. Initialize.
    xinfer::zoo::audio::SpeechRecognizer recognizer(config);

    // 3. Load a waveform of someone speaking.
    std::vector<float> speech_waveform; // Assume this is loaded
    
    // 4. Get the transcription.
    auto result = recognizer.predict(speech_waveform);

    // 5. Print the result.
    std::cout << "Transcription: \"" << result.text << "\"\n";
    std::cout << "Confidence: " << result.confidence << "\n";
}
```
**Config Struct:** `SpeechRecognizerConfig`
**Input:** `std::vector<float>` audio waveform.
**Output Struct:** `TranscriptionResult`.
**"F1 Car" Technology:** This pipeline uses the `postproc::ctc_decoder` to perform the complex CTC decoding on the GPU, avoiding a massive logits transfer to the CPU.

---

## `SpeakerIdentifier`

Identifies a person from a sample of their voice, based on a pre-registered database of known speakers.

**Header:** `#include <xinfer/zoo/audio/speaker_identifier.h>`

```cpp
#include <xinfer/zoo/audio/speaker_identifier.h>
#include <iostream>
#include <vector>

int main() {
    xinfer::zoo::audio::SpeakerIdentifierConfig config;
    config.engine_path = "assets/speaker_embedding.engine";

    xinfer::zoo::audio::SpeakerIdentifier identifier(config);

    // 1. Register known speakers by providing labeled voice samples.
    std::vector<float> alice_sample; // Load Alice's voice
    std::vector<float> bob_sample;   // Load Bob's voice
    identifier.register_speaker("Alice", alice_sample);
    identifier.register_speaker("Bob", bob_sample);
    std::cout << "Registered speakers: Alice, Bob\n";

    // 2. Load an unknown voice sample to identify.
    std::vector<float> unknown_sample; // Load an unknown voice
    
    // 3. Identify the speaker.
    auto result = identifier.identify(unknown_sample);

    // 4. Print the result.
    std::cout << "Identified speaker: " << result.speaker_label
              << " (Similarity: " << result.similarity_score << ")\n";
}
```
**Config Struct:** `SpeakerIdentifierConfig`
**Methods:**
- `register_speaker(const std::string&, const std::vector<float>&)` to enroll speakers.
- `identify(const std::vector<float>&)` to find the best match.
- `compare(const SpeakerEmbedding&, const SpeakerEmbedding&)` to get a similarity score between two embeddings.

---

## `EventDetector`

Detects the start and end times of specific sound events in a continuous audio stream.

**Header:** `#include <xinfer/zoo/audio/event_detector.h>`

```cpp
#include <xinfer/zoo/audio/event_detector.h>
#include <iostream>
#include <vector>

int main() {
    xinfer::zoo::audio::EventDetectorConfig config;
    config.engine_path = "assets/audio_event_detector.engine";
    config.labels_path = "assets/event_labels.txt"; // e.g., "glass_break", "siren"
    config.event_threshold = 0.7f;

    xinfer::zoo::audio::EventDetector detector(config);

    std::vector<float> long_audio_stream; // A long recording
    
    auto events = detector.predict(long_audio_stream);

    std::cout << "Detected " << events.size() << " audio events:\n";
    for (const auto& event : events) {
        printf(" - Event: %-15s, Start: %.2fs, End: %.2fs\n",
               event.label.c_str(), event.start_time_seconds, event.end_time_seconds);
    }
}
```
**Config Struct:** `EventDetectorConfig`
**Input:** `std::vector<float>` audio waveform.
**Output Struct:** `AudioEvent` (contains label, start/end times, and confidence).

---

## `MusicSourceSeparator`

Separates a mixed music track into its constituent sources, such as vocals, drums, bass, and other instruments.

**Header:** `#include <xinfer/zoo/audio/music_source_separator.h>`

```cpp
#include <xinfer/zoo/audio/music_source_separator.h>
#include <iostream>
#include <vector>
#include <map>

int main() {
    xinfer::zoo::audio::MusicSourceSeparatorConfig config;
    config.engine_path = "assets/music_separator.engine";
    config.source_names = {"vocals", "drums", "bass", "other"};

    xinfer::zoo::audio::MusicSourceSeparator separator(config);

    std::vector<float> song_waveform; // Load a full song
    
    // The result is a map from the source name to its isolated audio waveform.
    std::map<std::string, xinfer::zoo::audio::AudioWaveform> sources = separator.predict(song_waveform);

    std::cout << "Separated music into " << sources.size() << " sources:\n";
    for (const auto& pair : sources) {
        std::cout << " - Found source: " << pair.first << " (" << pair.second.size() << " samples)\n";
        // In a real app, you would save each waveform to a new .wav file.
    }
}
```
**Config Struct:** `MusicSourceSeparatorConfig`
**Input:** `std::vector<float>` mixed audio waveform.
**Output:** `std::map<std::string, AudioWaveform>`.