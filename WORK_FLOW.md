No problem. Since Mermaid diagrams can sometimes be hard to render or read depending on your Markdown viewer, I will translate those diagrams into **ASCII Art** and **Text-Based Workflows**.

Here are the three views of `xInfer` in a plain-text format.

---

### 1. The Architecture Stack (Layer View)

This diagram represents **"Who includes Who"**. High-level components sit on top of low-level components. A module can only see what is *below* it.

```text
===============================================================
                LAYER 5: USER APPLICATIONS
---------------------------------------------------------------
 [ xInfer Studio (UI) ]    [ xinfer-cli ]    [ Model Server ]
          |                      |                   |
          +----------------------+-------------------+
                            |
                            v
===============================================================
                LAYER 4: THE ZOO (Business Logic)
---------------------------------------------------------------
 [ Vision ]   [ Audio ]   [ NLP ]   [ Robotics ]   [ Science ]
      |           |          |           |              |
      +-----------+----------+-----------+--------------+
                            |
                            v
===============================================================
           LAYER 3: THE THREE PILLARS (Abstraction)
---------------------------------------------------------------
 [ Pre-Processing ]    [ Inference Backend ]    [ Post-Processing ]
 (CUDA, RGA, NEON)      (TRT, RKNN, Vitis)       (NMS, Decoding)
        |                       |                        |
        +-----------------------+------------------------+
                            |
                            v
===============================================================
                LAYER 2: CORE (Data Types)
---------------------------------------------------------------
 [ Tensor ]   [ Logging ]   [ Device Manager ]   [ Utils ]
                            |
                            v
===============================================================
                LAYER 1: HARDWARE SDKS (External)
---------------------------------------------------------------
 [ NVIDIA CUDA ]  [ Rockchip RKNPU ]  [ Intel OpenVINO ]  [ AMD Vitis ]
```

**Key Takeaway:** The `Zoo` orchestrates the "Three Pillars". The `Core` holds the data structures used by everyone.

---

### 2. The Runtime Workflow (Data Flow)

This represents **"What happens when you run `predict()`"**. Data flows from Left to Right.

```text
STEP 1: INPUT
   +----------------+
   |  Raw Image/AV  |  (cv::Mat, vector<float>)
   +----------------+
          |
          v
STEP 2: PRE-PROCESSING (Hardware Accelerated)
   +-------------------------------------------------------+
   | Module: src/preproc/                                  |
   | Action: Resize, Normalize, Colorspace, Layout (HWC->NCHW) |
   | Memory: Uploads data from Host RAM -> GPU/NPU Memory  |
   +-------------------------------------------------------+
          |
          v
STEP 3: INFERENCE (The "Engine")
   +-------------------------------------------------------+
   | Module: src/backends/                                 |
   | Action: Execute Neural Network (Forward Pass)         |
   | Memory: Reads/Writes directly on Device Memory (Zero-Copy) |
   +-------------------------------------------------------+
          |
          v
STEP 4: POST-PROCESSING (Decoding)
   +-------------------------------------------------------+
   | Module: src/postproc/                                 |
   | Action: Decode, NMS, Softmax, ArgMax                  |
   | Memory: Downloads small result list to CPU            |
   +-------------------------------------------------------+
          |
          v
STEP 5: RESULT
   +----------------+
   | Struct Objects |  (BoundingBox, ClassResult, Text)
   +----------------+
```

---

### 3. Concrete Example: Aegis Sky (FPGA Drone Tracking)

This shows specifically how the **Aegis Sky** module connects the pieces for a real-world task.

```text
[ CAMERA SOURCE ]
      |
      | (Raw Frame)
      v
+-------------------------------------------------------+
| ZOO MODULE: Aegis Sky (src/zoo/drones/aegis.cpp)      |
|                                                       |
|  1. Calls PREPROC                                     |
|     -> Uses "FpgaImagePreprocessor"                   |
|     -> Writes image to physical address 0xA0000000    |
|                                                       |
|  2. Calls BACKEND                                     |
|     -> Uses "VitisBackend"                            |
|     -> Tells FPGA DPU to read from 0xA0000000         |
|     -> DPU writes raw heatmap to 0xB0000000           |
|                                                       |
|  3. Calls POSTPROC                                    |
|     -> Uses "Detection3DPostproc"                     |
|     -> Reads 0xB0000000                               |
|     -> Calculates 3D coordinates (x, y, z)            |
|                                                       |
|  4. Internal Logic                                    |
|     -> Uses "CpuTracker" (Kalman Filter)              |
|     -> Assigns ID #42 to the detection                |
|                                                       |
+-------------------------------------------------------+
      |
      | (Tracked Object ID #42)
      v
[ FIRE CONTROL SYSTEM / GUI ]
```

### Summary of Relationships

1.  **Core** is the foundation. It contains `xinfer::core::Tensor`. Everyone includes Core.
2.  **Backends**, **Preproc**, and **Postproc** are siblings. They do not depend on each other. They only depend on `Core`.
3.  **Zoo** is the manager. It includes headers from Preproc, Backend, and Postproc to stitch them together into a useful application.
4.  **UI/CLI** sits at the very top. It only talks to `Zoo` (to run things) or `Compiler` (to build things).




---
# PART 2




You are absolutely right. The previous diagrams showed the "Core Engine," but the **Enterprise Modules** (`Serving`, `Flow`, `Telemetry`) sit **on top** or **alongside** that engine to make it usable in production.

Here are the updated schematics integrating these new components.

---

### 1. The Enterprise Architecture Stack (Updated)

This view shows how the new modules wrap the core engine.

*   **Flow & Serving** are **Application Wrappers**. They replace the need for you to write `main.cpp`.
*   **Telemetry** is a **Sidecar**. It sits alongside the processing pillars to watch them.

```text
========================================================================
                  LAYER 6: EXTERNAL WORLD
------------------------------------------------------------------------
 [ Web Browser ]      [ Prometheus/Grafana ]      [ JSON Config File ]
       |                        ^                          |
       | HTTP                   | Metrics                  | Load
       v                        |                          v
========================================================================
                  LAYER 5: ENTERPRISE RUNTIMES
------------------------------------------------------------------------
 [ xinfer::serving ]    [ xinfer::telemetry ]      [ xinfer::flow ]
 (Model Server)         (System Monitor)           (Pipeline Engine)
       |                        |                          |
       +-----------+            |             +------------+
                   |            v             |
                   v    (Monitors Perf)       v
========================================================================
                  LAYER 4: THE ZOO (Business Logic)
------------------------------------------------------------------------
 [ Vision ]   [ Audio ]   [ NLP ]   [ Robotics ]   [ Science ] ...
       |           |          |           |              |
       +-----------+----------+-----------+--------------+
                             |
                             v
========================================================================
            LAYER 3: THE THREE PILLARS (Abstraction)
------------------------------------------------------------------------
 [ Pre-Processing ]    [ Inference Backend ]     [ Post-Processing ]
                             |
                             v
========================================================================
            LAYER 2: CORE & HARDWARE
------------------------------------------------------------------------
 [ Tensor Memory ]  -->  [ NVIDIA / Rockchip / Intel / FPGA Hardware ]
```

---

### 2. The "Low-Code" Workflow (Using `xinfer::flow`)

This is what happens when you run `./xinfer-app --pipeline config.json`. The **Flow** module acts as the "Puppet Master."

```text
CONFIG FILE (pipeline.json)
      |
      v
[ FLOW ORCHESTRATOR ]
      | 1. Parses JSON
      | 2. Instantiates Nodes (Source, Zoo, Sink)
      | 3. Starts Loop
      |
      +---> NODE A: CameraSource
      |       | (Produces Packet)
      |       v
      +---> NODE B: ObjectDetector (Wraps zoo/vision/detector)
      |       | 
      |       +---> Calls Preproc -> Backend -> Postproc
      |       |
      |       v
      +---> NODE C: DisplaySink
              | (Visualizes Packet)
              v
           [ SCREEN ]
```

---

### 3. The "Microservice" Workflow (Using `xinfer::serving`)

This is what happens when a Python script sends a request to your C++ server.

```text
[ Python Client ]
      |
      | POST /v1/models/yolo:predict (JSON)
      v
[ SERVING MODULE ]
      |
      +---> 1. RequestHandler parses JSON
      |
      +---> 2. ModelRepository looks up Model
      |        (Lazy loads 'yolo.engine' if not ready)
      |
      +---> 3. Backend Execution
      |        (Runs inference on GPU/NPU)
      |
      +---> 4. Serialization
      |        (Converts Output Tensor -> JSON)
      |
      v
[ HTTP Response ]
```

---

### 4. The Telemetry Loop (The Observer)

This runs in a **Background Thread**, invisible to the main application logic, ensuring the system doesn't crash or overheat.

```text
[ MAIN THREAD (Inference) ]            [ TELEMETRY THREAD ]
           |                                     |
           | Start Prediction                    | (Wakes up every 1s)
           |                                     |
           |                                     +---> 1. Check Hardware
           |                                     |    (Read /proc/stat, NVML)
           |                                     |    (Get Temp, RAM Usage)
           |                                     |
           |                                     +---> 2. Check Drift
           |                                     |    (Compare current input
           |                                     |     stats vs baseline)
           |                                     |
           | End Prediction                      |
           +------------------------------------>| 3. Record Latency
                                                 |
                                                 v
                                        [ Exporter (File/HTTP) ]
```

### How to choose which one to use?

1.  **Building a Desktop App (GUI)?** Use **Core + Zoo + UI**. You write C++ code to call the Zoo directly.
2.  **Building a Smart Camera (Edge)?** Use **Flow**. You write a JSON file to define the pipeline (Cam -> Detect -> Network Sink). No C++ recompilation needed for logic changes.
3.  **Building a Cloud Service / API?** Use **Serving**. You spin up the server, and other apps talk to it via HTTP.
4.  **Running in Production?** Always enable **Telemetry** alongside any of the above.