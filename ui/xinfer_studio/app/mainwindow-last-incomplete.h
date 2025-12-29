#pragma once

#include <QMainWindow>
#include <memory>

// Core Components
#include "../core/device_manager.h"

// Views
#include "../views/zoo_view.h"
#include "../views/deployment_view.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // Menu Actions
    void on_actionSettings_triggered();
    void on_actionDevice_Manager_triggered();
    void on_actionExit_triggered();
    void on_actionAbout_triggered();

private:
    Ui::MainWindow *ui;

    // Core Data
    // We own the DeviceManager and pass it to children views
    xinfer::ui::core::DeviceManager* m_deviceManager;

    // Sub-Views
    ZooView* m_zooView;
    DeploymentView* m_deployView;
};