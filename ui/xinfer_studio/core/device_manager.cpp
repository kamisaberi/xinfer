#include "device_manager.h"
#include <QDebug>

namespace xinfer::ui::core {

DeviceManager::DeviceManager(const QString& jsonPath) : m_jsonPath(jsonPath) {
    // Try to load immediately on construction
    if (!load()) {
        qWarning() << "No existing devices.json found, starting empty.";
    }
}

QList<DeviceInfo>& DeviceManager::getDevices() {
    return m_devices;
}

const QList<DeviceInfo>& DeviceManager::getDevices() const {
    return m_devices;
}

void DeviceManager::addDevice(const DeviceInfo& device) {
    m_devices.append(device);
    save();
}

void DeviceManager::removeDevice(int index) {
    if (index >= 0 && index < m_devices.size()) {
        m_devices.removeAt(index);
        save();
    }
}

bool DeviceManager::load() {
    QFile file(m_jsonPath);
    if (!file.open(QIODevice::ReadOnly)) return false;

    QByteArray data = file.readAll();
    QJsonDocument doc = QJsonDocument::fromJson(data);

    if (doc.isNull() || !doc.isObject()) return false;

    QJsonObject root = doc.object();
    QJsonArray arr = root["devices"].toArray();

    m_devices.clear();
    for (const auto& val : arr) {
        QJsonObject obj = val.toObject();
        DeviceInfo dev;
        dev.name = obj["name"].toString();
        dev.ip = obj["ip"].toString();
        dev.user = obj["user"].toString();
        dev.target = obj["target"].toString();
        dev.path = obj["path"].toString();
        m_devices.append(dev);
    }
    return true;
}

bool DeviceManager::save() {
    QJsonArray arr;
    for (const auto& dev : m_devices) {
        QJsonObject obj;
        obj["name"] = dev.name;
        obj["ip"] = dev.ip;
        obj["user"] = dev.user;
        obj["target"] = dev.target;
        obj["path"] = dev.path;
        arr.append(obj);
    }

    QJsonObject root;
    root["devices"] = arr;

    QJsonDocument doc(root);
    QFile file(m_jsonPath);
    if (!file.open(QIODevice::WriteOnly)) return false;

    file.write(doc.toJson());
    return true;
}

} // namespace xinfer::ui::core