# Zoo API: Reinforcement Learning

The `xinfer::zoo::rl` module provides the core, high-performance building block for deploying agents trained with Reinforcement Learning (RL).

In RL, the inference step—running the **policy** network to decide on an action—is often in the "hot loop" of a real-time system. The latency of this single operation can determine the success or failure of the entire application, whether it's a robot, a trading bot, or a game AI.

The `xInfer` RL zoo is designed around a single, powerful, and generic class: the `Policy`.

---

## Core Component: `rl::Policy`

**Header:** `#include <xinfer/zoo/rl/policy.h>`

The `Policy` class is a hyper-optimized, low-latency engine for executing a trained RL agent's decision-making network. It is a generic wrapper that can run any policy that takes a state tensor as input and produces an action tensor as output.

The real "F1 car" magic comes from the TensorRT engine it loads. You would train your agent in a Python framework (like Stable Baselines3 or CleanRL), export the final actor/policy network to ONNX, and then use the `xinfer-cli` to build a hyper-optimized, INT8- or FP16-quantized engine. The `Policy` class then runs this engine with minimal overhead.

### Core API

```cpp
#include <xinfer/zoo/rl/policy.h>

// Configuration
struct PolicyConfig {
    std::string engine_path;
};

class Policy {
public:
    explicit Policy(const PolicyConfig& config);

    // Get an action for a single state
    core::Tensor predict(const core::Tensor& state);
    
    // Get actions for a batch of states (highly efficient)
    core::Tensor predict_batch(const core::Tensor& state_batch);
};
```

---

## Domain-Specific Applications

While the `rl::Policy` is generic, it serves as the core engine for many specialized, real-world applications. The following `zoo` classes are powerful examples of how to use the `Policy` engine to solve domain-specific problems.

### Industrial Robotics: `robotics::AssemblyPolicy`

**Header:** `#include <xinfer/zoo/robotics/assembly_policy.h>`

**Use Case:** Controls a robot arm to perform complex, vision-based assembly tasks. The policy takes a camera image and the robot's joint angles as input and outputs motor commands.

```cpp
// This function would be in the robot's 100Hz control loop
void execute_robot_step(xinfer::zoo::robotics::AssemblyPolicy& policy) {
    // 1. Get current state from sensors
    cv::Mat camera_image = get_camera_frame();
    std::vector<float> joint_states = get_joint_angles();

    // 2. Execute the policy to get the next action in milliseconds
    std::vector<float> next_action = policy.predict(camera_image, joint_states);

    // 3. Send action to motor controllers
    send_motor_commands(next_action);
}
```

### Autonomous Drones: `drones::NavigationPolicy`

**Header:** `#include <xinfer/zoo/drones/navigation_policy.h>`

**Use Case:** Enables agile, GPS-denied flight in cluttered environments. The policy takes a depth image and the drone's current state (velocity, orientation) as input and outputs flight control commands (roll, pitch, yaw, thrust).

```cpp
// This function runs inside the drone's flight controller
void navigate_step(xinfer::zoo::drones::NavigationPolicy& policy) {
    // 1. Get state from sensors
    cv::Mat depth_image = get_depth_camera_frame();
    std::vector<float> drone_state = get_imu_data();

    // 2. Execute the policy to get flight commands
    xinfer::zoo::drones::NavigationAction action = policy.predict(depth_image, drone_state);

    // 3. Send commands to the motors
    set_motor_outputs(action.roll, action.pitch, action.yaw, action.thrust);
}
```

### High-Frequency Trading: `hft::OrderExecutionPolicy`

**Header:** `#include <xinfer/zoo/hft/order_execution_policy.h>`

**Use Case:** Manages the execution of a large financial order to minimize market impact. The policy takes the current state of the limit order book as input and decides whether to place a small buy/sell order in the next microsecond.

```cpp
// This function is in the hot path of a trading application's event loop
void on_market_data_update(xinfer::zoo::hft::OrderExecutionPolicy& policy) {
    // 1. Get the current market state as a GPU tensor
    xinfer::core::Tensor market_state_tensor = get_order_book_tensor();

    // 2. Execute the policy with microsecond latency
    xinfer::zoo::hft::OrderExecutionAction action = policy.predict(market_state_tensor);

    // 3. Execute the trade
    if (action.action == OrderActionType::PLACE_BUY) {
        execute_buy_order(action.volume, action.price_level);
    }
}
```

### Game Development: `gaming::NPC_BehaviorPolicy`

**Header:** `#include <xinfer/zoo/gaming/npc_behavior_policy.h>

**Use Case:** Creates intelligent, non-scripted AI for hundreds of game characters. The policy takes a batch of states for all NPCs in a level and outputs a batch of actions.

```cpp
// This function runs once per game frame
void update_all_npc_ai(xinfer::zoo::gaming::NPCBehaviorPolicy& policy, World& world) {
    // 1. Gather the states of all active NPCs into a single batched tensor
    xinfer::core::Tensor npc_state_batch = world.get_all_npc_states();

    // 2. Execute the policy for all NPCs in a single, efficient GPU call
    xinfer::core::Tensor npc_action_batch = policy.predict_batch(npc_state_batch);

    // 3. Apply the actions to each NPC in the world
    world.set_all_npc_actions(npc_action_batch);
}
```