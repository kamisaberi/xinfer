#pragma once

#include <QDialog>
#include "../core/device_manager.h"

namespace Ui { class DeviceEditorDialog; }

class DeviceEditorDialog : public QDialog {
    Q_OBJECT

public:
    // Mode: Add New vs Edit Existing
    explicit DeviceEditorDialog(QWidget *parent = nullptr);
    explicit DeviceEditorDialog(const xinfer::ui::core::DeviceInfo& device, QWidget *parent = nullptr);
    ~DeviceEditorDialog();

    // Returns the struct populated from the UI fields
    xinfer::ui::core::DeviceInfo getDeviceInfo() const;

private slots:
    void on_btnSave_clicked();
    void on_btnCancel_clicked();

private:
    Ui::DeviceEditorDialog *ui;
    void setupUiData(const xinfer::ui::core::DeviceInfo& device);
};