#pragma once

#include <QMainWindow>
#include <QThread>
#include <QImage>
#include <QTimer>

// Internal Components
#include "../controllers/zoo_controller.h"
#include "../core/device_manager.h"
#include "../models/device_model.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // --- UI Interaction Slots (Auto-connected by name) ---
    void on_btnStart_clicked();
    void on_btnStop_clicked();
    void on_btnBrowse_clicked();

    // --- Menu Actions ---
    void on_actionDevice_Manager_triggered();
    void on_actionExit_triggered();

    // --- Worker Thread Slots ---
    /**
     * @brief Receive a processed frame from the backend.
     */
    void updateView(const QImage& image);

    /**
     * @brief Receive log messages from the backend.
     */
    void appendLog(QString message);

    /**
     * @brief Handle critical errors from the backend.
     */
    void handleError(QString message);

private:
    Ui::MainWindow *ui;

    // Data Models
    xinfer::ui::core::DeviceManager* m_deviceManager;
    xinfer::ui::models::DeviceModel* m_deviceModel;

    // Threading
    QThread* m_workerThread;
    ZooController* m_controller;

    // State
    bool m_isPipelineRunning = false;
    void updateUIState(bool running);
};