#pragma once

#include <QAbstractTableModel>
#include "../core/device_manager.h"

namespace xinfer::ui::models {

    class DeviceModel : public QAbstractTableModel {
        Q_OBJECT

    public:
        explicit DeviceModel(core::DeviceManager* manager, QObject* parent = nullptr);

        // Required Overrides
        int rowCount(const QModelIndex& parent = QModelIndex()) const override;
        int columnCount(const QModelIndex& parent = QModelIndex()) const override;
        QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
        QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

        // Helper to refresh UI when underlying data changes
        void refresh();

        // Get the full struct for a specific row
        core::DeviceInfo getDevice(int row) const;

    private:
        core::DeviceManager* m_manager;
    };

} // namespace xinfer::ui::models