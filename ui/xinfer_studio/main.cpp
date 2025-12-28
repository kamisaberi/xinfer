#include "app/mainwindow.h"

#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QDebug>

// Helper to load stylesheet
void apply_stylesheet(QApplication& app, const QString& path) {
    QFile file(path);
    if (file.open(QFile::ReadOnly | QFile::Text)) {
        QTextStream stream(&file);
        app.setStyleSheet(stream.readAll());
        file.close();
    } else {
        qWarning() << "Could not find stylesheet:" << path;
    }
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Set Application Metadata
    app.setApplicationName("xInfer Studio");
    app.setOrganizationName("xInfer");
    app.setApplicationVersion("1.0.0");

    // Load Dark Theme
    // We assume the qss file is copied to the build dir or in a resource file
    apply_stylesheet(app, ":/styles/dark_theme.qss");

    MainWindow w;
    w.show();

    return app.exec();
}