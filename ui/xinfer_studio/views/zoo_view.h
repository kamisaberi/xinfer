#pragma once

#include <QWidget>
#include <QThread>
#include "../controllers/zoo_controller.h"
#include "../widgets/video_display.h"
#include "../widgets/results_panel.h"

namespace Ui { class ZooView; }

class ZooView : public QWidget {
    Q_OBJECT

public:
    explicit ZooView(QWidget *parent = nullptr);
    ~ZooView();

private slots:
    void on_btnStart_clicked();
    void on_btnStop_clicked();

    // Updates from Controller
    void onFrameReady(const QImage& img);
    // In a real app, you'd emit results from controller too:
    // void onResultsReady(const std::vector<BoundingBox>& boxes);

private:
    Ui::ZooView *ui;

    QThread* m_workerThread;
    ZooController* m_controller;

    void setupThread();
};