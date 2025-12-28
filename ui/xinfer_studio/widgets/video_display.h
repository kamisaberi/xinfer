#pragma once
#include <QWidget>
#include <QImage>
#include <QPainter>

class VideoDisplay : public QWidget {
    Q_OBJECT
public:
    explicit VideoDisplay(QWidget *parent = nullptr);

public slots:
    void updateFrame(const QImage& image);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QImage m_currentFrame;
};