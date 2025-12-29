#include "stylesheet_loader.h"

#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QDebug>

namespace xinfer::ui::helpers {

    void apply_theme(QApplication& app, const QString& resourcePath) {
        QFile file(resourcePath);

        if (file.open(QFile::ReadOnly | QFile::Text)) {
            QTextStream stream(&file);
            QString styleSheet = stream.readAll();

            // Apply to the global application instance
            app.setStyleSheet(styleSheet);
            file.close();

            qInfo() << "[UI] Loaded theme successfully:" << resourcePath;
        } else {
            qWarning() << "[UI] Failed to load theme:" << resourcePath
                       << "-- Error:" << file.errorString();
        }
    }

} // namespace xinfer::ui::helpers