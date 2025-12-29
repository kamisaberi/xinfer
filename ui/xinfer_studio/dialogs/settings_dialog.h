#pragma once

#include <QDialog>
#include <QSettings>

namespace Ui { class SettingsDialog; }

class SettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = nullptr);
    ~SettingsDialog();

    // Static helper to get a setting anywhere in the app
    static QString getSetting(const QString& key, const QString& defaultValue = "");

private slots:
    void on_btnSave_clicked();
    void on_btnCancel_clicked();
    void on_btnBrowseSdk_clicked();

private:
    Ui::SettingsDialog *ui;
    void loadSettings();
};