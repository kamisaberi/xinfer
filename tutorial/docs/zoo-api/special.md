# Zoo API: Specialized & High-Value Models

The `xinfer::zoo::special` module is the home for our most advanced, domain-specific, and performance-critical solutions.

While other `zoo` modules provide optimized pipelines for common AI tasks, the classes in this module often solve problems that are **computationally infeasible** without a from-scratch, hyper-optimized C++/CUDA implementation. These are the "F1 car" engines for industries where every microsecond and every floating-point operation counts.

This module is for experts who are pushing the boundaries of what is possible with computational science and AI.

---

## `HFTModel` (High-Frequency Trading)

Provides a minimal-overhead, ultra-low-latency engine for executing a financial trading model.

**Header:** `#include <xinfer/zoo/special/hft.h>`

### Use Case: Microsecond-Scale Market Prediction

In high-frequency trading, the entire "perception-to-action" loop—from receiving a market data packet to sending an order—must happen in a few microseconds. The model inference is a critical part of this loop. The `HFTModel` is designed to be called from a low-level C++ trading application where performance is the only metric that matters.

```cpp
#include <xinfer/zoo/special/hft.h>
#include <xinfer/core/tensor.h>
#include <iostream>

// This function would be in the hot path of a trading application
void on_market_data_update(xinfer::zoo::special::HFTModel& model) {
    // 1. Receive market data (e.g., limit order book state)
    //    and place it directly into a pre-allocated GPU tensor.
    xinfer::core::Tensor market_state_tensor;
    // ... logic to populate the tensor from a network feed ...

    // 2. Execute the model to get a trading signal.
    //    This is a single, hyper-optimized TensorRT call.
    xinfer::zoo::special::TradingSignal signal = model.predict(market_state_tensor);

    // 3. Act on the signal immediately.
    if (signal.action == xinfer::zoo::special::TradingAction::BUY) {
        // ... execute buy order ...
    }
}

int main() {
    // 1. Configure and load the pre-compiled HFT policy engine.
    xinfer::zoo::special::HFTConfig config;
    config.engine_path = "assets/market_alpha_model.engine";
    
    xinfer::zoo::special::HFTModel model(config);

    // 2. Run in a tight loop.
    // while (market_is_open) {
    //     on_market_data_update(model);
    // }
    return 0;
}
```
**Config Struct:** `HFTConfig`
**Input:** `xinfer::core::Tensor` representing the current market state.
**Output Struct:** `TradingSignal` (an enum for BUY/SELL/HOLD and a confidence score).

---

## `FluidSimulator` (Physics Simulation)

Provides a real-time, GPU-accelerated fluid dynamics solver. This is not a neural network; it is a direct CUDA implementation of a physics model.

**Header:** `#include <xinfer/zoo/special/physics.h>`

### Use Case: Interactive VFX and Engineering

This class allows for real-time simulation of smoke, fire, or water for visual effects, or for interactive computational fluid dynamics (CFD) in engineering design.

```cpp
#include <xinfer/zoo/special/physics.h>
#include <xinfer/core/tensor.h>
#include <iostream>

int main() {
    // 1. Configure the simulator grid and fluid properties.
    xinfer::zoo::special::FluidSimulatorConfig config;
    config.resolution_x = 512;
    config.resolution_y = 512;
    config.viscosity = 0.01f;

    // 2. Initialize the simulator.
    xinfer::zoo::special::FluidSimulator simulator(config);

    // 3. Create GPU tensors to hold the state of the simulation.
    xinfer::core::Tensor velocity_field({1, 2, 512, 512}, xinfer::core::DataType::kFLOAT);
    xinfer::core::Tensor density_field({1, 1, 512, 512}, xinfer::core::DataType::kFLOAT);
    // ... initialize fields with some starting state ...

    // 4. Run the simulation loop.
    std::cout << "Running fluid simulation for 100 steps...\n";
    for (int i = 0; i < 100; ++i) {
        // Add a force or density source (e.g., from user input)
        // ...
        
        // This single call executes a chain of custom CUDA kernels (advect, diffuse, project).
        simulator.step(velocity_field, density_field);

        // (In a real app, you would render the density_field here)
    }
    std::cout << "Simulation complete.\n";
    return 0;
}
```
**Config Struct:** `FluidSimulatorConfig`
**Input/Output:** `xinfer::core::Tensor` objects representing the fluid's state, which are modified in-place.
**"F1 Car" Technology:** This class is a wrapper around a set of from-scratch, hyper-optimized CUDA kernels for solving the Navier-Stokes equations.

---

## `VariantCaller` (Genomics)

Provides an ultra-fast engine for genomic analysis, designed to work with next-generation sequencing data.

**Header:** `#include <xinfer/zoo/special/genomics.h>`

### Use Case: Accelerating Personalized Medicine

A researcher wants to analyze a patient's entire DNA sequence to find mutations linked to a disease. This requires a model that can process a sequence of billions of characters.

```cpp
#include <xinfer/zoo/special/genomics.h>
#include <iostream>
#include <string>

int main() {
    // 1. Configure the variant caller.
    //    The engine would be a hyper-optimized Mamba or long-convolution model.
    xinfer::zoo::special::VariantCallerConfig config;
    config.engine_path = "assets/genomic_foundation_model.engine";
    config.vocab_path = "assets/dna_vocab.json";

    // 2. Initialize.
    xinfer::zoo::special::VariantCaller caller(config);

    // 3. Load a long DNA sequence (e.g., an entire chromosome).
    std::string dna_sequence = "GATTACA..."; // This would be millions of characters long

    // 4. Run the prediction.
    //    This is only possible because the underlying Mamba engine can handle
    //    the massive sequence length with linear, not quadratic, complexity.
    std::cout << "Analyzing DNA sequence for variants...\n";
    std::vector<xinfer::zoo::special::GenomicVariant> variants = caller.predict(dna_sequence);

    std::cout << "Found " << variants.size() << " potential variants.\n";
    return 0;
}
```
**Config Struct:** `VariantCallerConfig`
**Input:** A `std::string` containing a very long DNA sequence.
**Output Struct:** `GenomicVariant` (contains the position, reference base, and alternate base).
**"F1 Car" Technology:** This class is built on top of a hyper-optimized **Mamba engine**, likely using a custom CUDA kernel for the selective scan operation. This is the only architecture that can feasibly handle the massive sequence lengths found in genomics.
