#include "zoo_controller.h"
#include "../helpers/qt_cv_converter.h"

using namespace xinfer;

ZooController::ZooController(QObject *parent) : QObject(parent) {}
ZooController::~ZooController() { stopPipeline(); }

void ZooController::startPipeline(const std::string& model_path, int camera_id) {
    if (m_running) return;

    // 1. Initialize Camera
    m_cap.open(camera_id);
    if (!m_cap.isOpened()) {
        emit errorOccurred("Failed to open camera.");
        return;
    }

    // 2. Initialize xInfer Engine (Heavy operation)
    try {
        zoo::vision::DetectorConfig config;
        config.target = Target::INTEL_OV; // Default for PC
        config.model_path = model_path;
        config.confidence_threshold = 0.5f;

        m_detector = std::make_unique<zoo::vision::ObjectDetector>(config);
    } catch (const std::exception& e) {
        emit errorOccurred(QString::fromStdString(e.what()));
        return;
    }

    m_running = true;

    // Start the loop immediately (0ms delay means run as fast as possible)
    QTimer::singleShot(0, this, &ZooController::processLoop);
}

void ZooController::stopPipeline() {
    m_running = false;
    if (m_cap.isOpened()) m_cap.release();
}

void ZooController::processLoop() {
    if (!m_running) return;

    cv::Mat frame;
    if (m_cap.read(frame)) {
        // --- xInfer Pipeline ---
        // 1. Inference
        auto detections = m_detector->predict(frame);

        // 2. Visualization (Draw boxes)
        for (const auto& det : detections) {
            cv::rectangle(frame, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, det.label, cv::Point(det.x1, det.y1-5), 1, 1.0, cv::Scalar(0,255,0));
        }

        // 3. Send to UI
        emit frameReady(ui::helpers::mat_to_qimage(frame));
    }

    // Schedule next frame
    QTimer::singleShot(10, this, &ZooController::processLoop);
}