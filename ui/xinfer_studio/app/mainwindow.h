#pragma once
#include <QMainWindow>
#include <QThread>
#include "../controllers/zoo_controller.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btnStart_clicked();
    void on_btnStop_clicked();
    void updateView(const QImage& image);
    void handleError(QString msg);

private:
    Ui::MainWindow *ui;

    // Threading
    QThread* m_workerThread;
    ZooController* m_controller;
};