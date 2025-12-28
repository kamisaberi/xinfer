#include "device_model.h"

namespace xinfer::ui::models {

    DeviceModel::DeviceModel(core::DeviceManager* manager, QObject* parent)
        : QAbstractTableModel(parent), m_manager(manager) {}

    int DeviceModel::rowCount(const QModelIndex&) const {
        return m_manager->getDevices().size();
    }

    int DeviceModel::columnCount(const QModelIndex&) const {
        return 4; // Name, IP, Target, User
    }

    QVariant DeviceModel::data(const QModelIndex& index, int role) const {
        if (!index.isValid()) return QVariant();

        const auto& device = m_manager->getDevices().at(index.row());

        if (role == Qt::DisplayRole) {
            switch (index.column()) {
                case 0: return device.name;
                case 1: return device.target;
                case 2: return device.ip;
                case 3: return device.user;
            }
        }
        return QVariant();
    }

    QVariant DeviceModel::headerData(int section, Qt::Orientation orientation, int role) const {
        if (role == Qt::DisplayRole && orientation == Qt::Horizontal) {
            switch (section) {
                case 0: return "Device Name";
                case 1: return "Platform";
                case 2: return "IP Address";
                case 3: return "Username";
            }
        }
        return QVariant();
    }

    void DeviceModel::refresh() {
        beginResetModel();
        endResetModel();
    }

    core::DeviceInfo DeviceModel::getDevice(int row) const {
        if (row >= 0 && row < m_manager->getDevices().size()) {
            return m_manager->getDevices().at(row);
        }
        return {};
    }

} // namespace xinfer::ui::models