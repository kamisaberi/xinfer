#pragma once
#include <QObject>
#include <QTimer>
#include <opencv2/opencv.hpp>

// Include the Zoo
#include <xinfer/zoo.h>

class ZooController : public QObject {
    Q_OBJECT
public:
    explicit ZooController(QObject *parent = nullptr);
    ~ZooController();

public slots:
    // Called by UI thread to start/stop
    void startPipeline(const std::string& model_path, int camera_id);
    void stopPipeline();

private slots:
    // Internal loop
    void processLoop();

    signals:
        // Emitted to UI
        void frameReady(const QImage& image);
    void errorOccurred(QString message);

private:
    bool m_running = false;
    cv::VideoCapture m_cap;

    // The Zoo Module (Polymorphic wrapper could be used here)
    std::unique_ptr<xinfer::zoo::vision::ObjectDetector> m_detector;
};