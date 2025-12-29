#include "deployment_view.h"
#include "./ui_deployment_view.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDateTime>

DeploymentView::DeploymentView(QWidget *parent)
    : QWidget(parent), ui(new Ui::DeploymentView)
{
    ui->setupUi(this);
    setupThread();
}

DeploymentView::~DeploymentView() {
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
    delete ui;
}

void DeploymentView::setupThread() {
    m_workerThread = new QThread(this);
    m_controller = new CompilerController();
    m_controller->moveToThread(m_workerThread);

    connect(m_controller, &CompilerController::logOutput, this, &DeploymentView::onLog);
    connect(m_controller, &CompilerController::progressChanged, this, &DeploymentView::onProgress);
    connect(m_controller, &CompilerController::compilationFinished, this, &DeploymentView::onFinished);

    m_workerThread->start();
}

void DeploymentView::setDeviceManager(xinfer::ui::core::DeviceManager* manager) {
    m_deviceManager = manager;
    populateTargets();
}

void DeploymentView::populateTargets() {
    ui->cmbTarget->clear();
    // Add standard targets
    ui->cmbTarget->addItem("NVIDIA TensorRT", "nv-trt");
    ui->cmbTarget->addItem("Rockchip RKNN", "rockchip-rknn");
    ui->cmbTarget->addItem("Intel OpenVINO", "intel-ov");
    ui->cmbTarget->addItem("AMD Vitis AI", "amd-vitis");
    ui->cmbTarget->addItem("Qualcomm QNN", "qcom-qnn");
    // ... add others or load dynamically from devices.json logic if preferred
}

void DeploymentView::on_btnBrowseInput_clicked() {
    QString path = QFileDialog::getOpenFileName(this, "Select Model", "", "ONNX Models (*.onnx);;TFLite Models (*.tflite)");
    if (!path.isEmpty()) {
        ui->lineEditInput->setText(path);

        // Auto-suggest output path
        QFileInfo fi(path);
        QString base = fi.path() + "/" + fi.completeBaseName();
        // Extension depends on target, just leave empty or default to .engine
        ui->lineEditOutput->setText(base + ".engine");
    }
}

void DeploymentView::on_btnBrowseOutput_clicked() {
    QString path = QFileDialog::getSaveFileName(this, "Save Engine As", ui->lineEditOutput->text());
    if (!path.isEmpty()) {
        ui->lineEditOutput->setText(path);
    }
}

void DeploymentView::on_btnCompile_clicked() {
    QString input = ui->lineEditInput->text();
    QString output = ui->lineEditOutput->text();
    QString target = ui->cmbTarget->currentData().toString();
    QString precision = ui->cmbPrecision->currentText().toLower();

    if (input.isEmpty() || output.isEmpty()) {
        QMessageBox::warning(this, "Error", "Input and Output paths are required.");
        return;
    }

    ui->btnCompile->setEnabled(false);
    ui->progressBar->setValue(0);
    ui->txtLog->clear();
    onLog("Starting build job...");

    QVariantMap params;
    // Add calibration path if UI has a field for it
    // params["calibration_data"] = ...

    // Invoke on background thread
    QMetaObject::invokeMethod(m_controller, "compileModel",
                              Qt::QueuedConnection,
                              Q_ARG(QString, target),
                              Q_ARG(QString, input),
                              Q_ARG(QString, output),
                              Q_ARG(QString, precision),
                              Q_ARG(QVariantMap, params));
}

void DeploymentView::onLog(QString msg) {
    QString ts = QDateTime::currentDateTime().toString("HH:mm:ss");
    ui->txtLog->appendPlainText(QString("[%1] %2").arg(ts, msg));
}

void DeploymentView::onProgress(int percent) {
    ui->progressBar->setValue(percent);
}

void DeploymentView::onFinished(bool success, QString msg) {
    ui->btnCompile->setEnabled(true);
    if (success) {
        QMessageBox::information(this, "Success", msg);
    } else {
        QMessageBox::critical(this, "Failed", msg);
    }
}