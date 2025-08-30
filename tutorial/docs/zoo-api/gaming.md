# Zoo API: Game Development

The `xinfer::zoo::gaming` module provides hyper-optimized solutions for the most demanding computational problems in modern video game development.

These tools are designed to be integrated into game engines like Unreal Engine and Unity to enable next-generation AI behaviors, physics simulations, and content creation workflows that are impossible to achieve with standard, general-purpose tools. In an industry where real-time performance is paramount, `xInfer` provides the critical F1-car components needed to build truly next-generation games.

---

## `NPCBehaviorPolicy`

A hyper-optimized engine for running thousands of individual AI "brains" for Non-Player Characters (NPCs) simultaneously.

**Header:** `#include <xinfer/zoo/gaming/npc_behavior_policy.h>`

### Use Case: Creating Smart, Emergent AI

Instead of simple, scripted AI, train complex reinforcement learning policies that allow NPCs to react intelligently to the player and the world. The `NPCBehaviorPolicy` engine is designed to run these policies for every NPC in a level within a single game frame.

```cpp
#include <xinfer/zoo/gaming/npc_behavior_policy.h>
#include <xinfer/core/tensor.h>

// This function would be called once per game frame from the main engine loop.
void update_all_npc_ai(xinfer::zoo::gaming::NPCBehaviorPolicy& policy, GameWorld& world) {
    // 1. Gather the current state of all active NPCs into a single batched tensor on the GPU.
    xinfer::core::Tensor npc_state_batch = world.get_all_npc_states_as_gpu_tensor();

    // 2. Execute the policy for all NPCs in a single, efficient GPU call.
    //    This is massively faster than running hundreds of individual model calls.
    xinfer::core::Tensor npc_action_batch = policy.predict_batch(npc_state_batch);

    // 3. Apply the resulting actions back to each NPC in the game world.
    world.set_all_npc_actions_from_gpu_tensor(npc_action_batch);
}

int main() {
    // Initialize the policy engine once during level load.
    xinfer::zoo::gaming::NPCBehaviorPolicyConfig config;
    config.engine_path = "assets/npc_behavior_policy.engine";

    xinfer::zoo::gaming::NPCBehaviorPolicy policy(config);

    // while (game_is_running) {
    //     update_all_npc_ai(policy, current_world);
    // }
}
```
**Config Struct:** `NPCBehaviorPolicyConfig`
**Input:** A batched `core::Tensor` where each row is the state of one NPC.
**Output:** A batched `core::Tensor` where each row is the action for one NPC.
**"F1 Car" Technology:** The key is **batched inference**. This engine is optimized to run hundreds or thousands of small MLP forward passes in a single GPU launch, avoiding the massive overhead of individual inference calls from a game engine script.

---

## `FluidSimulator` (from `zoo::special`)

A real-time, GPU-accelerated fluid dynamics solver for interactive water, smoke, and fire.

**Header:** `#include <xinfer/zoo/special/physics.h>`

### Use Case: Dynamic, Interactive Environments

Create game levels with truly dynamic water that reacts to characters, or realistic smoke and fire that fills rooms and curls around obstacles. This moves beyond static, pre-baked animations.

```cpp
#include <xinfer/zoo/special/physics.h>
#include <xinfer/core/tensor.h>

int main() {
    // Initialize the simulator once.
    xinfer::zoo::special::FluidSimulatorConfig config;
    config.resolution_x = 256;
    config.resolution_y = 256;
    xinfer::zoo::special::FluidSimulator simulator(config);

    // Create GPU tensors to hold the fluid state.
    xinfer::core::Tensor velocity_field, density_field;
    
    // In the game's physics update loop (the "tick").
    // while (game_is_running) {
    //     // Add forces/density to the simulation (e.g., from a player walking through water).
    //     add_player_forces_to_gpu_tensor(velocity_field);
        
    //     // This single call runs a chain of custom CUDA kernels.
    //     simulator.step(velocity_field, density_field);

    //     // Render the resulting density_field as smoke or water.
    //     render_gpu_tensor(density_field);
    // }
}
```
**"F1 Car" Technology:** This is not a neural network. It is a **from-scratch CUDA implementation** of a physics solver (like Smoothed-Particle Hydrodynamics). It is orders of magnitude faster than a CPU-based physics engine and more specialized than the engine's built-in rigid body physics.

---

## `LightBaker` (Conceptual Tool)

A hyper-optimized tool for pre-calculating complex global illumination and lighting, reducing multi-hour bakes to minutes.

### Use Case: Accelerating Artist Iteration

A lighting artist needs to see the results of their changes quickly. Waiting overnight for a light bake is a major workflow bottleneck. This tool provides near-instant feedback.

```cpp
// This would be a command-line tool or a plugin button in the editor.
#include <xinfer/tools/light_baker.h> // Conceptual header
#include <iostream>

int main(int argc, char** argv) {
    // 1. Configure the light baker.
    xinfer::zoo::tools::LightBakerConfig config;
    config.scene_file = argv; // e.g., "my_level.gltf"
    config.output_path = "my_level_lightmaps/";
    config.quality = "High";
    config.num_bounces = 8;

    // 2. Initialize and run the bake.
    xinfer::zoo::tools::LightBaker baker(config);
    
    std::cout << "Starting light bake...\n";
    baker.bake();
    std::cout << "Light bake complete. Results saved to " << config.output_path << std.endl;

    return 0;
}
```
**"F1 Car" Technology:** A **custom CUDA path tracing engine**, written from scratch. It is faster than a game engine's built-in baker because it is specialized *only* for baking static lightmaps and ignores all the complexity of real-time rendering.