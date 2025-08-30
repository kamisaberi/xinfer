# Zoo API: High-Frequency Trading (HFT)

The `xinfer::zoo::hft` module provides an ultra-low-latency, "zero-overhead" solution for executing AI-driven trading strategies.

**This is the most specialized module in the `xInfer` ecosystem.** It is designed for a single purpose: to provide the fastest possible path from market data to a trading decision. In the world of HFT, performance is not measured in milliseconds; it is measured in **microseconds and nanoseconds**.

This module is not a general-purpose trading framework. It is a set of hyper-optimized inference engines designed to be integrated into a sophisticated, co-located C++ trading system.

!!! danger "Extreme Performance, Expert Use"
    The classes in this module make critical trade-offs for speed. They bypass many standard software abstractions and assume they are being run in a dedicated, real-time environment. They are intended for expert users with a deep understanding of low-latency systems.

---

## `OrderExecutionPolicy`

Executes a trained Reinforcement Learning (RL) policy for optimal trade execution.

**Header:** `#include <xinfer/zoo/hft/order_execution_policy.h>`

### Use Case: Minimizing Market Impact

A major challenge in algorithmic trading is executing a large order (e.g., selling 1 million shares of a stock) without causing the price to move against you. An RL agent can be trained in a market simulator to learn an optimal "scheduling" policy—breaking the large order into many small "child" orders and placing them over time to minimize impact.

The `OrderExecutionPolicy` is the hyper-optimized engine for running this learned policy in a live market.

```cpp
#include <xinfer/zoo/hft/order_execution_policy.h>
#include <xinfer/core/tensor.h>
#include <iostream>

// This function would be in the critical path of a C++ trading application's event loop.
// It would be triggered by a new market data update.
void on_market_tick(xinfer::zoo::hft::OrderExecutionPolicy& policy, TradingSystem& system) {
    // 1. Get the current market state directly into a pre-allocated GPU tensor.
    //    In a real HFT system, this would use kernel-bypass networking to get
    //    data to the GPU with the lowest possible latency.
    xinfer::core::Tensor market_state_tensor = system.get_current_market_state_gpu();

    // 2. Execute the policy. This is a single, deterministic, microsecond-scale call.
    xinfer::zoo::hft::OrderExecutionAction action = policy.predict(market_state_tensor);

    // 3. Act on the decision immediately.
    if (action.action == xinfer::zoo::hft::OrderActionType::PLACE_SELL) {
        system.execute_sell_order(action.volume, action.price_level);
    }
}

int main() {
    // 1. Configure and load the pre-compiled policy engine during application startup.
    xinfer::zoo::hft::OrderExecutionPolicyConfig config;
    config.engine_path = "assets/vwap_execution_policy.engine";
    
    xinfer::zoo::hft::OrderExecutionPolicy policy(config);
    std::cout << "HFT Order Execution Policy loaded and ready.\n";
    
    // TradingSystem system;
    // system.connect_to_exchange();
    
    // 2. Run in a tight, pinned-thread event loop.
    // while (system.is_market_open()) {
    //     if (system.has_new_market_data()) {
    //         on_market_tick(policy, system);
    //     }
    // }

    return 0;
}
```
**Config Struct:** `OrderExecutionPolicyConfig`
**Input:** `xinfer::core::Tensor` representing the current state of the limit order book and other market features.
**Output Struct:** `OrderExecutionAction` (an enum for BUY/SELL/HOLD, plus volume and price level).
**"F1 Car" Technology:** This class wraps a TensorRT engine that has been built from a very small, simple MLP. The key is that the entire `xInfer` runtime has minimal overhead, allowing it to be called in a hard real-time loop where every nanosecond of jitter matters.

---

## `MarketDataParser`

This is not a model, but a hyper-performant utility. It's a custom CUDA kernel for parsing raw, high-frequency financial data feeds.

**Header:** `#include <xinfer/zoo/special/hft.h>`
*(Note: For simplicity, this could live in the same `hft.h` header)*

### Use Case: Bypassing the CPU for Market Data

The first bottleneck in any HFT system is parsing the stream of raw binary data from the exchange. Doing this on the CPU is slow. This `MarketDataParser` is designed to work with network cards that support **GPU Direct**, allowing network packets to be streamed directly into GPU memory.

```cpp
#include <xinfer/zoo/special/hft.h>
#include <xinfer/core/tensor.h>

int main() {
    // 1. Initialize the parser. This loads the custom CUDA kernel.
    xinfer::zoo::hft::MarketDataParser parser;

    // 2. In the network receive loop:
    //    - A raw network buffer is received directly into GPU memory.
    void* d_raw_packet_buffer = get_packet_from_network_card_gpu();
    size_t packet_size = 1024; // in bytes

    //    - A GPU tensor is pre-allocated to hold the structured order book.
    xinfer::core::Tensor market_state_tensor({1, 20, 2}, xinfer::core::DataType::kFLOAT); // e.g. 20 levels of bids/asks

    // 3. Launch the CUDA kernel to parse the raw data and update the state tensor.
    //    This is a single, fast kernel launch.
    parser.parse_and_update(d_raw_packet_buffer, packet_size, market_state_tensor);

    // `market_state_tensor` is now ready to be fed into a policy engine.
    
    return 0;
}
```
**"F1 Car" Technology:** This is the ultimate "F1 car" component. It is a **from-scratch CUDA kernel** that implements a parser for a specific financial protocol (like FIX/FAST or a proprietary exchange protocol). By doing this on the GPU, you completely eliminate the CPU from the critical path, which is a massive competitive advantage.