# Zoo API: Robotics

The `xinfer::zoo::robotics` module provides a suite of hyper-optimized, low-latency pipelines for common robotics tasks.

In robotics, performance is not optional. The "perception-to-action" loop—the time it takes for a robot to see the world, understand it, and react—must often be measured in milliseconds. The `zoo` classes in this module are designed from the ground up for these hard real-time constraints, providing the core building blocks for intelligent robotic systems.

These pipelines are built to run efficiently on power-constrained embedded hardware, such as the NVIDIA Jetson platform.

---

## `AssemblyPolicy`

Executes a trained Reinforcement Learning (RL) policy for complex, vision-based manipulation tasks.

**Header:** `#include <xinfer/zoo/robotics/assembly_policy.h>`

### Use Case: AI-Driven Robotic Assembly

This class is designed for tasks like peg-in-hole insertion, wire routing, or component assembly, where a robot must learn from both visual and physical feedback. The `AssemblyPolicy` acts as the robot's "brain," taking in sensor data and outputting low-level motor commands.

```cpp
#include <xinfer/zoo/robotics/assembly_policy.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// This function would be in the robot's main control loop, running at high frequency (e.g., 100Hz)
void execute_robot_step(xinfer::zoo::robotics::AssemblyPolicy& policy) {
    // 1. Get current state from the robot's sensors.
    cv::Mat camera_image; // Image from the robot's wrist-mounted camera
    std::vector<float> joint_states; // Current angles of the robot's joints
    // ... update camera_image and joint_states from the robot's hardware interface ...

    // 2. Execute the policy to get the next action.
    //    This is a single, ultra-low-latency call to a fused C++/TensorRT pipeline.
    std::vector<float> next_action = policy.predict(camera_image, joint_states);

    // 3. Send the action to the robot's motor controllers.
    // ... send motor commands based on next_action ...
}

int main() {
    // 1. Configure and initialize the assembly policy.
    //    This loads two pre-built engines: one for vision, one for the policy MLP.
    xinfer::zoo::robotics::AssemblyPolicyConfig config;
    config.vision_encoder_engine_path = "assets/robot_vision_encoder.engine";
    config.policy_engine_path = "assets/peg_insertion_policy.engine";

    xinfer::zoo::robotics::AssemblyPolicy policy(config);
    std::cout << "Assembly policy loaded and ready.\n";

    // 2. Run the robot's control loop.
    // while (task_is_active) {
    //     execute_robot_step(policy);
    // }

    return 0;
}
```
**Config Struct:** `AssemblyPolicyConfig`
**Input:** `cv::Mat` from a camera and a `std::vector<float>` of the robot's physical state.
**Output:** `std::vector<float>` representing the next motor commands (e.g., joint velocities or end-effector deltas).
**"F1 Car" Technology:** This class orchestrates multiple, hyper-optimized TensorRT engines (a vision encoder and a policy MLP) within a hard real-time C++ application, ensuring a consistent, low-latency perception-to-action loop.

---

## `GraspPlanner`

Performs 6D pose estimation on an object to determine a stable grasp for a robotic hand.

**Header:** `#include <xinfer/zoo/robotics/grasp_planner.h>`
*(Note: This is a conceptual name for a 6D Pose Estimator specialized for robotics)*

### Use Case: Bin Picking

A robot needs to pick a specific object out of a bin of jumbled, randomly oriented parts. It must accurately determine the object's 3D position and orientation to grasp it successfully.

```cpp
#include <xinfer/zoo/robotics/grasp_planner.h> // Conceptual header
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Initialize the grasp planner.
    //    The engine would be a specialized model trained on 3D data.
    xinfer::zoo::robotics::GraspPlannerConfig config;
    config.engine_path = "assets/6d_pose_estimator.engine";

    xinfer::zoo::robotics::GraspPlanner planner(config);

    // 2. Get an RGB-D image from a 3D camera.
    cv::Mat color_image; // ... from sensor
    cv::Mat depth_image; // ... from sensor

    // 3. Predict the 6D pose of the target object.
    auto grasp_poses = planner.predict(color_image, depth_image);

    // 4. Execute the grasp with the robot arm.
    if (!grasp_poses.empty()) {
        std::cout << "Found a graspable object. Executing pick.\n";
        // ... robot control logic to move to the grasp_poses ...
    } else {
        std::cout << "No graspable object found.\n";
    }

    return 0;
}
```
**Config Struct:** `GraspPlannerConfig`
**Input:** `cv::Mat` for color and depth data.
**Output:** A list of possible 6D grasp poses (position + quaternion rotation).
**"F1 Car" Technology:** This pipeline would use custom CUDA kernels for point cloud processing (e.g., from the depth image) and a fused TensorRT engine for the final pose regression.
