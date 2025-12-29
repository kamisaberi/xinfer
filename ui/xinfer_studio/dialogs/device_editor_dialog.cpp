#include "device_editor_dialog.h"
#include "./ui_device_editor_dialog.h"
#include <QMessageBox>

DeviceEditorDialog::DeviceEditorDialog(QWidget *parent)
    : QDialog(parent), ui(new Ui::DeviceEditorDialog)
{
    ui->setupUi(this);
    setWindowTitle("Add New Device");
}

DeviceEditorDialog::DeviceEditorDialog(const xinfer::ui::core::DeviceInfo& device, QWidget *parent)
    : QDialog(parent), ui(new Ui::DeviceEditorDialog)
{
    ui->setupUi(this);
    setWindowTitle("Edit Device");
    setupUiData(device);
}

DeviceEditorDialog::~DeviceEditorDialog() {
    delete ui;
}

void DeviceEditorDialog::setupUiData(const xinfer::ui::core::DeviceInfo& device) {
    ui->lineEditName->setText(device.name);
    ui->lineEditIp->setText(device.ip);
    ui->lineEditUser->setText(device.user);
    ui->lineEditPath->setText(device.path);

    // Set ComboBox to match target
    int idx = ui->cmbTarget->findData(device.target);
    if (idx >= 0) ui->cmbTarget->setCurrentIndex(idx);
}

xinfer::ui::core::DeviceInfo DeviceEditorDialog::getDeviceInfo() const {
    xinfer::ui::core::DeviceInfo info;
    info.name = ui->lineEditName->text();
    info.ip = ui->lineEditIp->text();
    info.user = ui->lineEditUser->text();
    info.path = ui->lineEditPath->text();
    info.target = ui->cmbTarget->currentData().toString();
    return info;
}

void DeviceEditorDialog::on_btnSave_clicked() {
    if (ui->lineEditName->text().isEmpty() || ui->lineEditIp->text().isEmpty()) {
        QMessageBox::warning(this, "Validation Error", "Name and IP Address are required.");
        return;
    }
    accept();
}

void DeviceEditorDialog::on_btnCancel_clicked() {
    reject();
}