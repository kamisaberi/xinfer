#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::education {

    enum class TutorState {
        IDLE,
        ASKING,
        LISTENING,
        EVALUATING,
        RESPONDING
    };

    struct TutorInteraction {
        TutorState state;
        std::string tutor_speech;      // Text the tutor is saying
        std::string student_transcript;  // What the student said
        bool is_answer_correct;
        float answer_score;
    };

    struct TutorConfig {
        // Hardware Target (Edge CPU/NPU for local interaction)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Models ---
        // 1. ASR (Speech-to-Text)
        std::string asr_model_path;

        // 2. Grader (Semantic Similarity)
        std::string grader_model_path;

        // 3. Feedback Generator (Small LLM)
        std::string feedback_model_path;

        // 4. TTS (Text-to-Speech)
        std::string tts_model_path;

        // --- Tokenizers/Vocabs ---
        std::string asr_vocab_path;
        std::string grader_vocab_path;
        std::string feedback_vocab_path;
        std::string tts_vocab_path;

        // --- Lesson Plan ---
        // Map: Question ID -> {Question Text, Model Answer}
        std::map<std::string, std::pair<std::string, std::string>> lesson_plan;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Tutor {
    public:
        explicit Tutor(const TutorConfig& config);
        ~Tutor();

        // Move semantics
        Tutor(Tutor&&) noexcept;
        Tutor& operator=(Tutor&&) noexcept;
        Tutor(const Tutor&) = delete;
        Tutor& operator=(const Tutor&) = delete;

        /**
         * @brief Start the session and ask the first question.
         * @return Interaction state with the first question.
         */
        TutorInteraction start();

        /**
         * @brief Process a chunk of student's speech audio.
         *
         * @param pcm_chunk Raw float audio samples.
         * @return The current interaction state (e.g., updated transcript).
         */
        TutorInteraction on_audio(const std::vector<float>& pcm_chunk);

        /**
         * @brief Signal that the student has finished speaking.
         * Triggers evaluation and feedback generation.
         * @return The final evaluation and the tutor's response.
         */
        TutorInteraction on_student_finished();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::education