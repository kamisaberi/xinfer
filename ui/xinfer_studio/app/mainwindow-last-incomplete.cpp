#include "mainwindow.h"
#include "./ui_mainwindow.h"

// Dialogs
#include "../dialogs/settings_dialog.h"
#include "../dialogs/device_editor_dialog.h" // If you have a manager dialog, or use the editor directly

#include <QMessageBox>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 1. Initialize Core Services
    m_deviceManager = new xinfer::ui::core::DeviceManager("devices.json");

    // 2. Initialize Views
    m_zooView = new ZooView(this);
    m_deployView = new DeploymentView(this);

    // 3. Inject Dependencies
    // The Deployment View needs the Device Manager to populate the target list
    m_deployView->setDeviceManager(m_deviceManager);
    // The Zoo View might need it later for remote inference
    // m_zooView->setDeviceManager(m_deviceManager);

    // 4. Setup Tab Widget
    // We remove any placeholder tabs from Designer and add our real views
    ui->mainTabWidget->clear();
    ui->mainTabWidget->addTab(m_zooView, QIcon(":/icons/run.svg"), "Inference Lab");
    ui->mainTabWidget->addTab(m_deployView, QIcon(":/icons/deploy.svg"), "Deployment Center");

    // Status Bar
    ui->statusbar->showMessage("Ready. Loaded " + QString::number(m_deviceManager->getDevices().size()) + " devices.");
}

MainWindow::~MainWindow() {
    delete m_deviceManager;
    delete ui;
}

// =============================================================================
// Menu Slots
// =============================================================================

void MainWindow::on_actionSettings_triggered() {
    SettingsDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        ui->statusbar->showMessage("Settings saved.", 3000);
    }
}

void MainWindow::on_actionDevice_Manager_triggered() {
    // In a full app, you might have a dedicated DeviceManagerDialog that lists devices
    // and lets you add/edit/delete. For now, let's just open the "Add Device" dialog
    // as a quick shortcut, or implement the list logic here.

    // Example: Launching the Editor for a new device
    DeviceEditorDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        auto newDev = dlg.getDeviceInfo();
        m_deviceManager->addDevice(newDev);

        // Notify views to refresh
        // m_deployView->refreshDevices();

        ui->statusbar->showMessage("Device added: " + newDev.name, 3000);
    }
}

void MainWindow::on_actionExit_triggered() {
    close();
}

void MainWindow::on_actionAbout_triggered() {
    QMessageBox::about(this, "About xInfer Studio",
        "<h3>xInfer Studio v1.0</h3>"
        "<p>The Universal AI Deployment & Inference Toolkit.</p>"
        "<p>Supports NVIDIA, Intel, AMD, Rockchip, Qualcomm, and more.</p>"
        "<p>(C) 2025 xInfer Team</p>");
}