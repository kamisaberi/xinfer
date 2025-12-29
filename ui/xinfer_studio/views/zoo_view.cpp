#include "zoo_view.h"
#include "./ui_zoo_view.h" // You create this via Qt Designer similar to mainwindow.ui

ZooView::ZooView(QWidget *parent)
    : QWidget(parent), ui(new Ui::ZooView)
{
    ui->setupUi(this);
    setupThread();
}

ZooView::~ZooView() {
    QMetaObject::invokeMethod(m_controller, "stopPipeline");
    m_workerThread->quit();
    m_workerThread->wait();
    delete ui;
}

void ZooView::setupThread() {
    m_workerThread = new QThread(this);
    m_controller = new ZooController();
    m_controller->moveToThread(m_workerThread);

    connect(m_controller, &ZooController::frameReady, this, &ZooView::onFrameReady);
    // Connect error signals, etc.

    m_workerThread->start();
}

void ZooView::on_btnStart_clicked() {
    QString path = ui->lineEditModel->text();
    // Start Logic...
    QMetaObject::invokeMethod(m_controller, "startPipeline", Q_ARG(std::string, path.toStdString()), Q_ARG(int, 0));
}

void ZooView::on_btnStop_clicked() {
    QMetaObject::invokeMethod(m_controller, "stopPipeline");
}

void ZooView::onFrameReady(const QImage& img) {
    ui->videoDisplay->updateFrame(img);
}