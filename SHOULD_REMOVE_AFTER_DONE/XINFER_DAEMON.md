md
You are absolutely right. To build **xFabric** (the OS), we first need a rock-solid **xInfer Daemon (`xinferd`)** (the Kernel) to manage the hardware and execution state.

The `xinferd` daemon is not just a script; it is a **long-running background service** that maintains memory state, handles hardware locking (e.g., reserving the FPGA or NPU), and exposes a control API.

Here is the **architectural structure** and **module definition** for the `xInfer` Daemon.

### 1. The Directory Structure
We will create a dedicated directory `src/daemon` (or upgrade your existing `src/serving`). This separates the *library logic* (`src/core`, `src/backends`) from the *service logic*.

```text
xinfer/
└── src/
    └── daemon/
        ├── CMakeLists.txt
        ├── main.cpp                  # Entry point (./xinferd)
        ├── config.json               # Default daemon config (port, log level)
        │
        ├── core/                     # The Brain
        │   ├── service.cpp           # Lifecycle (Init, Start, Stop, Signal Handling)
        │   ├── session_manager.cpp   # Manages loaded models (The "RAM" of the OS)
        │   ├── resource_monitor.cpp  # Watchdog for GPU/NPU Temps & VRAM
        │   └── worker_pool.cpp       # Thread pool for handling concurrent requests
        │
        ├── api/                      # The Interface (How xFabric talks to it)
        │   ├── http_server.cpp       # REST API (Command & Control)
        │   ├── grpc_server.cpp       # (Optional) High-perf streaming
        │   ├── routes/
        │   │   ├── control_routes.cpp  # /load, /unload, /configure
        │   │   ├── infer_routes.cpp    # /predict (Binary/JSON)
        │   │   └── system_routes.cpp   # /telemetry, /health
        │   └── serializers.cpp       # JSON <-> C++ Structs
        │
        └── ipc/                      # Zero-Copy Optimization
            └── shm_manager.cpp       # Shared Memory handling (for local xFabric video feeds)
```

---

### 2. Key Modules & Responsibilities

Here are the specific C++ classes that make up the Daemon.

#### A. The Session Manager (`core/session_manager.h`)
This is the most critical module. It turns `xInfer` from a "script runner" into a "platform." It holds the state of loaded engines so they don't need to be re-initialized for every request.

*   **Role:** Maintains a `std::unordered_map` of active models.
*   **Key Functionality:** Hot-swapping. It allows `xFabric` to unload a "Day Mode" model and load a "Night Mode" model without restarting the service.

```cpp
// Conceptual C++ Structure
class SessionManager {
public:
    // Returns a UUID for the loaded session
    std::string loadModel(const std::string& modelPath, TargetDevice device);
    
    // Frees VRAM/FPGA resources
    void unloadModel(const std::string& sessionID);
    
    // The hot path: retrieves the pre-warmed engine
    std::shared_ptr<xinfer::Engine> getEngine(const std::string& sessionID);

private:
    // Thread-safe map of ID -> Active Engine
    std::unordered_map<std::string, std::shared_ptr<xinfer::Engine>> active_sessions_;
    std::mutex session_mutex_;
};
```

#### B. The API Server (`api/http_server.h`)
This defines the "System Calls" of your AI OS. `xFabric` (the UI/Orchestrator) uses these endpoints to control the daemon.

*   **Technology:** `httplib` (from your `third_party` folder).
*   **Endpoints:**
    1.  `POST /v1/control/load`: Tells daemon to compile/load a model from the Zoo.
    2.  `POST /v1/infer/{id}`: Sends image data, returns JSON detections.
    3.  `GET /v1/telemetry`: Returns `{ "gpu_temp": 65, "fps": 120, "vram_usage": "2GB" }`.
    4.  `GET /v1/device/capabilities`: Auto-detects if NPU/FPGA/CUDA is available.

#### C. The Resource Monitor (`core/resource_monitor.h`)
This module connects to the low-level drivers. It prevents the system from crashing by rejecting new models if VRAM is full.

*   **Role:** Periodically polls hardware stats.
*   **Integration:**
    *   **NVIDIA:** Uses `NVML`.
    *   **Rockchip:** Parses `/sys/class/devfreq/dmc/load`.
    *   **FPGA:** Queries Xilinx Runtime (XRT).

#### D. The IPC Layer (`ipc/shm_manager.h`)
*Critical for 4K Video.*
If `xFabric` (GUI) and `xinferd` (Daemon) are on the same machine, sending video over HTTP (localhost) is too slow.
*   **Role:** Creates a named Shared Memory block (e.g., `/dev/shm/xinfer_video_buffer`).
*   **Flow:**
    1.  `xFabric` writes raw frame to SHM.
    2.  `xFabric` sends `POST /infer` with `{"shm_handle": "frame_01"}`.
    3.  `xinferd` reads SHM (Zero-Copy), runs inference, writes result to SHM.

---

### 3. Implementation Plan

To establish this Daemon structure, I recommend we implement the **Session Manager** and the **Basic HTTP Server** first. This will allow your Qt UI (`xFabric`) to actually "connect" to something.

**Shall I generate the code for `src/daemon/main.cpp` and `src/daemon/core/session_manager.cpp` to get the service running?**