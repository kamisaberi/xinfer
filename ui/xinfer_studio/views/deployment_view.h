#pragma once

#include <QWidget>
#include "../controllers/compiler_controller.h"
#include "../core/device_manager.h" // To populate target list

namespace Ui { class DeploymentView; }

class DeploymentView : public QWidget {
    Q_OBJECT

public:
    explicit DeploymentView(QWidget *parent = nullptr);
    ~DeploymentView();

    // Inject dependencies
    void setDeviceManager(xinfer::ui::core::DeviceManager* manager);

private slots:
    void on_btnBrowseInput_clicked();
    void on_btnBrowseOutput_clicked();
    void on_btnCompile_clicked();

    // Updates from Controller
    void onLog(QString msg);
    void onProgress(int percent);
    void onFinished(bool success, QString msg);

private:
    Ui::DeploymentView *ui;

    QThread* m_workerThread;
    CompilerController* m_controller;
    xinfer::ui::core::DeviceManager* m_deviceManager;

    void setupThread();
    void populateTargets();
};