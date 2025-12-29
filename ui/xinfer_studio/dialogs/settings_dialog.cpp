#include "settings_dialog.h"
#include "./ui_settings_dialog.h"
#include <QFileDialog>

SettingsDialog::SettingsDialog(QWidget *parent)
    : QDialog(parent), ui(new Ui::SettingsDialog)
{
    ui->setupUi(this);
    loadSettings();
}

SettingsDialog::~SettingsDialog() {
    delete ui;
}

void SettingsDialog::loadSettings() {
    QSettings settings("xInfer", "Studio");
    ui->lineEditSdkPath->setText(settings.value("paths/sdk_root", "").toString());
    ui->checkAutoConnect->setChecked(settings.value("app/auto_connect", false).toBool());
}

void SettingsDialog::on_btnSave_clicked() {
    QSettings settings("xInfer", "Studio");
    settings.setValue("paths/sdk_root", ui->lineEditSdkPath->text());
    settings.setValue("app/auto_connect", ui->checkAutoConnect->isChecked());
    accept();
}

void SettingsDialog::on_btnCancel_clicked() {
    reject();
}

void SettingsDialog::on_btnBrowseSdk_clicked() {
    QString dir = QFileDialog::getExistingDirectory(this, "Select SDK Root");
    if (!dir.isEmpty()) ui->lineEditSdkPath->setText(dir);
}

QString SettingsDialog::getSetting(const QString& key, const QString& defaultValue) {
    QSettings settings("xInfer", "Studio");
    return settings.value(key, defaultValue).toString();
}