#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::gaming {

    /**
     * @brief Sensory inputs for the NPC agent.
     */
    struct NpcObservation {
        // Player/Self State
        float health_percent;
        bool has_weapon;
        int ammo_count;

        // Environment State
        // e.g., raycast distances to walls/obstacles
        std::vector<float> raycast_results;

        // Target State
        bool is_enemy_visible;
        float enemy_distance;
        float enemy_angle_rad; // Relative angle to enemy
    };

    /**
     * @brief Discrete actions the NPC can take.
     */
    enum class NpcAction {
        IDLE = 0,
        MOVE_FORWARD = 1,
        STRAFE_LEFT = 2,
        STRAFE_RIGHT = 3,
        TURN_LEFT = 4,
        TURN_RIGHT = 5,
        ATTACK = 6,
        TAKE_COVER = 7
    };

    struct PolicyResult {
        NpcAction action;
        float confidence;
    };

    struct NpcPolicyConfig {
        // Hardware Target (CPU is usually sufficient for single-agent RL policies)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., ppo_shooter_agent.onnx)
        std::string model_path;

        // Input Specs (Size of the flattened observation vector)
        int observation_dim = 16;

        // Action space size (Number of output logits)
        int action_dim = 8;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class NpcBehaviorPolicy {
    public:
        explicit NpcBehaviorPolicy(const NpcPolicyConfig& config);
        ~NpcBehaviorPolicy();

        // Move semantics
        NpcBehaviorPolicy(NpcBehaviorPolicy&&) noexcept;
        NpcBehaviorPolicy& operator=(NpcBehaviorPolicy&&) noexcept;
        NpcBehaviorPolicy(const NpcBehaviorPolicy&) = delete;
        NpcBehaviorPolicy& operator=(const NpcBehaviorPolicy&) = delete;

        /**
         * @brief Get the next action based on the game state.
         *
         * @param obs The NPC's current perception of the world.
         * @return The decided action.
         */
        PolicyResult get_action(const NpcObservation& obs);

        /**
         * @brief Reset internal state (for RNN-based policies).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::gaming