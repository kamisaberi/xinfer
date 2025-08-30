# Zoo API: Autonomous Drones

The `xinfer::zoo::drones` module provides hyper-optimized, low-latency pipelines for the core perception and control tasks required for autonomous drones.

In aerial robotics, the "perception-to-action" loop is paramount. Decisions must be made in milliseconds to ensure stable, reactive flight, especially in cluttered, GPS-denied environments. The `zoo` classes in this module are designed to be the "brain stem" of a drone's flight controller, providing the essential AI capabilities with the hard real-time performance that C++ and `xInfer` guarantee.

These pipelines are built to run on power-constrained, embedded NVIDIA Jetson hardware, which is the standard for the drone industry.

---

## `NavigationPolicy`

Executes a trained Reinforcement Learning (RL) policy for autonomous, vision-based navigation.

**Header:** `#include <xinfer/zoo/drones/navigation_policy.h>`

### Use Case: Obstacle Avoidance in a Cluttered Environment

A drone needs to fly through a dense forest or a warehouse to reach a target. It cannot rely on GPS. Instead, it uses a forward-facing depth camera to perceive its surroundings. At every step, an RL policy takes the depth image and the drone's current velocity as input, and outputs the optimal flight commands to avoid obstacles and make progress towards its goal.

This entire control loop must run at a very high frequency (e.g., 50-100Hz) to ensure the drone can react to obstacles in time.

```cpp
#include <xinfer/zoo/drones/navigation_policy.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// This function represents the drone's main, high-frequency control loop.
void flight_control_step(xinfer::zoo::drones::NavigationPolicy& policy, DroneHardware& drone) {
    // 1. Get the latest sensor readings from the drone's hardware.
    cv::Mat depth_image = drone.get_depth_camera_frame();
    std::vector<float> drone_state = drone.get_imu_and_velocity_data();

    // 2. Execute the policy to get the next flight command.
    //    This is a single, ultra-low-latency call to the TensorRT engine.
    //    The entire operation should take only a few milliseconds.
    xinfer::zoo::drones::NavigationAction action = policy.predict(depth_image, drone_state);

    // 3. Send the low-level motor commands to the flight controller.
    drone.set_motor_commands(action.roll, action.pitch, action.yaw, action.thrust);
}

int main() {
    // 1. Configure and initialize the navigation policy during drone startup.
    //    The engine is pre-built from a policy (typically a CNN+MLP)
    //    trained in a photorealistic simulator like NVIDIA Isaac Sim.
    xinfer::zoo::drones::NavigationPolicyConfig config;
    config.engine_path = "assets/forest_navigation_policy.engine";

    xinfer::zoo::drones::NavigationPolicy policy(config);
    std::cout << "Drone navigation policy loaded and ready.\n";
    
    // DroneHardware drone; // A placeholder for the actual drone's hardware interface
    
    // 2. Run the main flight control loop.
    // while (drone.is_armed()) {
    //     flight_control_step(policy, drone);
    //     // The loop would be timed to run at a fixed frequency (e.g., 100Hz).
    // }

    return 0;
}
```
**Config Struct:** `NavigationPolicyConfig`
**Input:** `cv::Mat` depth image and a `std::vector<float>` of the drone's physical state.
**Output Struct:** `NavigationAction` (contains roll, pitch, yaw, and thrust commands).
**"F1 Car" Technology:** The policy's neural network (typically a small CNN to process the image, followed by an MLP) is compiled by `xInfer`'s `builders` into a single, fused, INT8-quantized TensorRT engine. This allows the entire perception-to-action pipeline to run with the minimal latency and power consumption that are critical for extending a drone's flight time and ensuring its stability.
