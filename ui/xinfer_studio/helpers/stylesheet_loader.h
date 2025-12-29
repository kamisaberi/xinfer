#pragma once
#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QDebug>

namespace xinfer::ui::helpers {

    inline void apply_theme(QApplication& app, const QString& resourcePath) {
        QFile file(resourcePath);
        if (file.open(QFile::ReadOnly | QFile::Text)) {
            QTextStream stream(&file);
            app.setStyleSheet(stream.readAll());
            file.close();
            qInfo() << "Loaded theme:" << resourcePath;
        } else {
            qWarning() << "Failed to load theme:" << resourcePath;
        }
    }

}