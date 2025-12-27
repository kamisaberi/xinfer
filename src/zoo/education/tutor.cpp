#include <xinfer/zoo/education/tutor.h>
#include <xinfer/core/logging.h>

// --- We compose other Zoo modules ---
#include <xinfer/zoo/audio/speech_recognizer.h>
#include <xinfer/zoo/education/grader.h>
#include <xinfer/zoo/nlp/text_generator.h>
#include <xinfer/zoo/generative/text_to_speech.h>

#include <iostream>
#include <string>
#include <vector>

namespace xinfer::zoo::education {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Tutor::Impl {
    TutorConfig config_;

    // --- Components ---
    std::unique_ptr<audio::SpeechRecognizer> asr_;
    std::unique_ptr<education::Grader> grader_;
    std::unique_ptr<nlp::TextGenerator> feedback_generator_;
    std::unique_ptr<generative::TextToSpeech> tts_;

    // --- State ---
    TutorState current_state_ = TutorState::IDLE;
    std::string current_question_id_;
    std::string current_transcript_ = "";
    std::vector<std::string> lesson_keys_;
    int lesson_idx_ = 0;

    Impl(const TutorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Init ASR
        audio::SpeechRecognizerConfig asr_cfg;
        asr_cfg.target = config_.target;
        asr_cfg.model_path = config_.asr_model_path;
        asr_cfg.vocab_path = config_.asr_vocab_path;
        asr_ = std::make_unique<audio::SpeechRecognizer>(asr_cfg);

        // 2. Init Grader
        GraderConfig grader_cfg;
        grader_cfg.target = config_.target;
        grader_cfg.similarity_model_path = config_.grader_model_path;
        grader_cfg.similarity_vocab_path = config_.grader_vocab_path;
        grader_ = std::make_unique<education::Grader>(grader_cfg);

        // 3. Init Feedback LLM
        nlp::TextGenConfig llm_cfg;
        llm_cfg.target = config_.target;
        llm_cfg.model_path = config_.feedback_model_path;
        llm_cfg.tokenizer_path = config_.feedback_vocab_path;
        llm_cfg.max_new_tokens = 50;
        feedback_generator_ = std::make_unique<nlp::TextGenerator>(llm_cfg);

        // 4. Init TTS
        generative::TtsConfig tts_cfg;
        tts_cfg.target = config_.target;
        tts_cfg.acoustic_model_path = config_.tts_model_path; // Assuming split models
        // tts_cfg.vocoder_model_path = ...
        tts_ = std::make_unique<generative::TextToSpeech>(tts_cfg);

        // Populate lesson keys
        for(const auto& kv : config_.lesson_plan) {
            lesson_keys_.push_back(kv.first);
        }
    }

    TutorInteraction get_current_interaction() {
        TutorInteraction i;
        i.state = current_state_;
        i.student_transcript = current_transcript_;
        return i;
    }
};

// =================================================================================
// Public API
// =================================================================================

Tutor::Tutor(const TutorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Tutor::~Tutor() = default;
Tutor::Tutor(Tutor&&) noexcept = default;
Tutor& Tutor::operator=(Tutor&&) noexcept = default;

TutorInteraction Tutor::start() {
    if (!pimpl_) throw std::runtime_error("Tutor is null.");

    // Pick first question
    pimpl_->lesson_idx_ = 0;
    pimpl_->current_question_id_ = pimpl_->lesson_keys_[0];

    auto interaction = pimpl_->get_current_interaction();
    interaction.state = TutorState::ASKING;
    interaction.tutor_speech = pimpl_->config_.lesson_plan[pimpl_->current_question_id_].first;

    // TODO: Synthesize speech via TTS and play it
    // auto audio_res = pimpl_->tts_->synthesize(interaction.tutor_speech);

    // Transition to listening
    pimpl_->current_state_ = TutorState::LISTENING;
    pimpl_->current_transcript_ = "";

    return interaction;
}

TutorInteraction Tutor::on_audio(const std::vector<float>& pcm_chunk) {
    if (!pimpl_ || pimpl_->current_state_ != TutorState::LISTENING) {
        return pimpl_->get_current_interaction();
    }

    // Real-time transcription
    // Note: ASR zoo module needs a streaming `process_chunk` for this.
    // For this example, we assume it can be called repeatedly.
    auto asr_results = pimpl_->asr_->recognize(pcm_chunk);
    if (!asr_results.empty()) {
        pimpl_->current_transcript_ += asr_results[0] + " ";
    }

    return pimpl_->get_current_interaction();
}

TutorInteraction Tutor::on_student_finished() {
    if (!pimpl_ || pimpl_->current_state_ != TutorState::LISTENING) {
        return pimpl_->get_current_interaction();
    }

    pimpl_->current_state_ = TutorState::EVALUATING;
    auto interaction = pimpl_->get_current_interaction();

    // 1. Grade the answer
    RubricItem rubric;
    rubric.question_id = pimpl_->current_question_id_;
    rubric.model_answer = pimpl_->config_.lesson_plan[pimpl_->current_question_id_].second;
    rubric.type = QuestionType::SHORT_ANSWER_TEXT;

    auto grade = pimpl_->grader_->grade_text(rubric, pimpl_->current_transcript_);

    interaction.is_answer_correct = grade.is_correct;
    interaction.answer_score = grade.score;

    // 2. Generate Feedback with LLM
    pimpl_->current_state_ = TutorState::RESPONDING;

    std::string prompt = "Question: " + rubric.model_answer + "\n";
    prompt += "Student Answer: " + pimpl_->current_transcript_ + "\n";
    prompt += "Feedback: ";
    if (grade.is_correct) {
        prompt += "Correct! Explain why in a friendly, encouraging way.";
    } else {
        prompt += "Incorrect. Gently explain the mistake and provide a hint.";
    }

    interaction.tutor_speech = pimpl_->feedback_generator_->generate(prompt);

    // TODO: Synthesize feedback via TTS
    // pimpl_->tts_->synthesize(interaction.tutor_speech);

    // For now, reset to Idle. In a real app, might wait for TTS to finish.
    pimpl_->current_state_ = TutorState::IDLE;

    return interaction;
}

} // namespace xinfer::zoo::education