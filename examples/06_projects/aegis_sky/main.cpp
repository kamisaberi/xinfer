#include <iostream>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    // Target: AMD FPGA (Ultra-Low Latency)
    Target target = Target::AMD_VITIS;

    // 1. Setup Inference
    auto engine = backends::BackendFactory::create(target);
    engine->load_model("centerpoint_vitis.xmodel");

    // 2. Setup 3D Post-Processing (Decode Heatmaps -> Boxes)
    auto detector = postproc::create_detection3d(target);
    postproc::Detection3DConfig det_cfg;
    det_cfg.voxel_size_x = 0.1f;
    det_cfg.voxel_size_y = 0.1f;
    det_cfg.score_threshold = 0.35f;
    detector->init(det_cfg);

    // 3. Setup Tracker (Kalman Filter)
    auto tracker = postproc::create_tracker(target);
    postproc::TrackerConfig trk_cfg;
    trk_cfg.max_age = 10; // Keep track alive for 10 frames if detection missed
    tracker->init(trk_cfg);

    core::Tensor heatmaps, regression;

    // 4. Radar/Lidar Loop
    while (true) {
        // ... (Code to get data from sensor into tensors) ...

        // Inference
        engine->predict({/*input pointcloud*/}, {heatmaps, regression});

        // Detect
        std::vector<postproc::BoundingBox3D> dets = detector->process({heatmaps, regression});

        // Track
        // Converts stateless detections into persistent IDs
        std::vector<postproc::TrackedObject> tracks = tracker->update(dets); // Adapters needed for Box3D->Box

        // Action Logic
        for (const auto& t : tracks) {
            std::cout << "Target ID: " << t.track_id
                      << " | Dist: " << t.box.z << "m"
                      << " | Vel: (" << t.velocity_x << ", " << t.velocity_y << ")"
                      << std::endl;

            if (t.box.z < 50.0f) {
                std::cout << ">>> ENGAGE TARGET " << t.track_id << " <<<" << std::endl;
            }
        }
    }
}