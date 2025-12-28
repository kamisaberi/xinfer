#pragma once
#include <QImage>
#include <opencv2/opencv.hpp>

namespace xinfer::ui::helpers {
    /**
     * @brief Converts cv::Mat to QImage efficiently (zero-copy if possible).
     */
    QImage mat_to_qimage(const cv::Mat& mat);
}