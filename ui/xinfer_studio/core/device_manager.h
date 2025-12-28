#pragma once

#include <QString>
#include <QList>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>

namespace xinfer::ui::core {

    struct DeviceInfo {
        QString name;
        QString ip;
        QString user;
        QString target; // e.g. "nv-trt", "rockchip-rknn"
        QString path;   // Remote path
    };

    class DeviceManager {
    public:
        explicit DeviceManager(const QString& jsonPath = "devices.json");

        bool load();
        bool save();

        QList<DeviceInfo>& getDevices();
        const QList<DeviceInfo>& getDevices() const;

        void addDevice(const DeviceInfo& device);
        void removeDevice(int index);

    private:
        QString m_jsonPath;
        QList<DeviceInfo> m_devices;
    };

} // namespace xinfer::ui::core