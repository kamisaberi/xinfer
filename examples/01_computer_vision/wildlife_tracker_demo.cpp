#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <third_party/cxxopts/cxxopts.hpp> // Assuming this is the CLI parser

// The only header you need to include for the entire Zoo!
#include <xinfer/zoo.h>

using namespace xinfer;

int main(int argc, char** argv) {
    cxxopts::Options options("wildlife_tracker", "xInfer Demo: Detect & Track Animals in Video");
    options.add_options()
        ("t,target", "Hardware target (e.g., nv-trt, intel-ov, rockchip-rknn)", cxxopts::value<std::string>()->default_value("intel-ov"))
        ("m,model", "Path to the compiled detection model (.engine, .xml, .rknn)", cxxopts::value<std::string>())
        ("l,labels", "Path to the labels file (e.g., coco.names)", cxxopts::value<std::string>())
        ("v,video", "Path to the input video file", cxxopts::value<std::string>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);
    if (result.count("help") || !result.count("model") || !result.count("video")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // -------------------------------------------------------------------------
    // 1. CONFIGURE THE ZOO MODULE
    // -------------------------------------------------------------------------
    try {
        zoo::vision::AnimalTrackerConfig config;

        // --- The Magic of xInfer ---
        // This single line determines whether to use CUDA, OpenVINO, RKNN, etc.
        // The factories will handle everything else automatically.
        config.target = compiler::stringToTarget(result["target"].as<std::string>());

        config.model_path = result["model"].as<std::string>();
        config.labels_path = result["labels"].as<std::string>();

        // Set which classes from the model we care about (COCO animal classes)
        config.filter_class_ids = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24}; // cat, dog, horse... bear, zebra

        // Lower confidence for better tracking recall
        config.conf_threshold = 0.4f;

        // -------------------------------------------------------------------------
        // 2. INITIALIZE THE TRACKER
        // -------------------------------------------------------------------------
        std::cout << "Initializing Animal Tracker for target: " << result["target"].as<std::string>() << "..." << std::endl;
        zoo::vision::AnimalTracker tracker(config);

        // -------------------------------------------------------------------------
        // 3. RUN THE VIDEO LOOP
        // -------------------------------------------------------------------------
        cv::VideoCapture cap(result["video"].as<std::string>());
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file." << std::endl;
            return -1;
        }

        cv::Mat frame;
        int frame_count = 0;

        while (cap.read(frame)) {
            frame_count++;
            auto start_time = std::chrono::high_resolution_clock::now();

            // --- The Core xInfer Call ---
            // This single function call runs the entire pipeline:
            // Preprocess -> Detect -> NMS -> Track -> Scale
            auto tracks = tracker.track(frame);

            auto end_time = std::chrono::high_resolution_clock::now();
            float fps = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            // --- 4. VISUALIZE ---
            for (const auto& animal : tracks) {
                cv::Rect box(animal.x1, animal.y1, animal.x2 - animal.x1, animal.y2 - animal.y1);
                cv::Scalar color = cv::Scalar((animal.track_id * 50) % 255, (animal.track_id * 90) % 255, (animal.track_id * 120) % 255);

                cv::rectangle(frame, box, color, 2);

                std::string label = "ID:" + std::to_string(animal.track_id) + " " + animal.species;
                cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
            }

            cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            cv::imshow("xInfer Wildlife Tracker", frame);
            if (cv::waitKey(1) == 27) break; // ESC to quit
        }

        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}