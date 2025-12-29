#include "app/mainwindow.h"
#include "helpers/stylesheet_loader.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    // High DPI Scaling for modern monitors
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QApplication app(argc, argv);

    // Application Metadata
    app.setApplicationName("xInfer Studio");
    app.setOrganizationName("xInfer");
    app.setApplicationVersion("1.0.0");

    // Load Dark Theme
    // Ensure resources.qrc is added to CMakeLists.txt
    xinfer::ui::helpers::apply_theme(app, ":/styles/dark_theme.qss");

    MainWindow w;
    w.show();

    return app.exec();
}