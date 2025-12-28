#include "video_display.h"

VideoDisplay::VideoDisplay(QWidget *parent) : QWidget(parent) {
    // Set a dark background
    setAttribute(Qt::WA_OpaquePaintEvent);
    setStyleSheet("background-color: #1e1e1e;");
}

void VideoDisplay::updateFrame(const QImage& image) {
    m_currentFrame = image;
    update(); // Schedules a repaint
}

void VideoDisplay::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (m_currentFrame.isNull()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No Signal");
        return;
    }

    // Scale aspect ratio
    QImage scaled = m_currentFrame.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Center image
    int x = (width() - scaled.width()) / 2;
    int y = (height() - scaled.height()) / 2;

    painter.drawImage(x, y, scaled);
}