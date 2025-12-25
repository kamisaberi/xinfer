#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

// Thread-safe Queue for Pipelining
template<typename T>
class SafeQueue {
    std::queue<T> q;
    std::mutex m;
    std::condition_variable c;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(m);
        q.push(val);
        c.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lock(m);
        c.wait(lock, [this]{ return !q.empty(); });
        T val = q.front();
        q.pop();
        return val;
    }
};

struct FrameContext {
    int id;
    cv::Mat raw_image;
    core::Tensor input_tensor;
    core::Tensor output_tensor;
    std::vector<postproc::BoundingBox> results;
};

int main() {
    // 1. Setup Vitis AI Engine with Multiple Runners
    // "NUM_RUNNERS=3" tells xInfer to instantiate 3 independent DPU contexts.
    // This allows 3 CPU threads to call predict() simultaneously without blocking.
    Target target = Target::AMD_VITIS;

    auto engine = backends::BackendFactory::create(target);
    Config backend_cfg;
    backend_cfg.model_path = "yolov8_kv260.xmodel";
    backend_cfg.vendor_params = { "NUM_RUNNERS=3", "DPU_ARCH=kria_arch.json" };
    engine->load_model(backend_cfg.model_path);

    // Setup Processors (Lightweight CPU versions)
    auto preproc = preproc::create_image_preprocessor(target);
    preproc->init({640, 640, preproc::ImageFormat::RGB});

    auto postproc = postproc::create_detection(target);
    postproc->init({0.45f, 0.45f});

    // Queues for Pipeline Stages
    SafeQueue<FrameContext*> q_pre_to_infer;
    SafeQueue<FrameContext*> q_infer_to_post;

    // 2. Thread: Inference Worker
    // This runs in parallel with the camera capture
    std::thread t_inference([&]() {
        while (true) {
            FrameContext* ctx = q_pre_to_infer.pop();
            if (ctx->id == -1) break; // Exit signal

            // DPU Execution (Non-blocking if other runners are free)
            engine->predict({ctx->input_tensor}, {ctx->output_tensor});

            q_infer_to_post.push(ctx);
        }
    });

    // 3. Thread: Post-processing & Display
    std::thread t_post([&]() {
        while (true) {
            FrameContext* ctx = q_infer_to_post.pop();
            if (ctx->id == -1) break;

            // CPU NMS
            ctx->results = postproc->process({ctx->output_tensor});

            // Visualization (Simplified)
            std::cout << "Frame " << ctx->id << ": Detected " << ctx->results.size() << " objects." << std::endl;

            // Cleanup manually allocated context
            delete ctx;
        }
    });

    // 4. Main Thread: Capture & Preprocess
    cv::VideoCapture cap("drone_feed.mp4");
    int frame_id = 0;

    while (true) {
        FrameContext* ctx = new FrameContext();
        ctx->id = frame_id++;

        if (!cap.read(ctx->raw_image)) {
            delete ctx;
            break;
        }

        // Preprocess (Resize/Normalize) on CPU/FPGA-PL
        // This happens while the DPU is busy processing the previous frame
        preproc::process({ctx->raw_image.data, ctx->raw_image.cols, ctx->raw_image.rows, preproc::ImageFormat::BGR},
                         ctx->input_tensor);

        q_pre_to_infer.push(ctx);
    }

    // Shutdown
    FrameContext* stop_sig = new FrameContext(); stop_sig->id = -1;
    q_pre_to_infer.push(stop_sig); // Push once, relay logic needed in real app or join

    t_inference.join();
    // q_infer_to_post.push(stop_sig); // In real app handle signaling properly
    t_post.detach(); // Lazy exit for example

    return 0;
}