#include "qt_cv_converter.h"

namespace xinfer::ui::helpers {

    QImage mat_to_qimage(const cv::Mat& mat) {
        if (mat.empty()) return QImage();

        // 1. Handle Grayscale
        if (mat.type() == CV_8UC1) {
            // Must copy because QImage requires 32-bit alignment per line usually
            QImage img(mat.data, mat.cols, mat.rows, (int)mat.step, QImage::Format_Grayscale8);
            return img.copy(); // Deep copy to detach from Mat memory
        }

        // 2. Handle RGB/BGR
        if (mat.type() == CV_8UC3) {
            // OpenCV is BGR, Qt is RGB. We swap channels.
            // Option A: cv::cvtColor (Slower but safe)
            // Option B: QImage::Format_BGR888 (Fastest)

            QImage img(mat.data, mat.cols, mat.rows, (int)mat.step, QImage::Format_BGR888);
            return img.copy(); // Deep copy to ensure UI doesn't crash if Mat is deleted
        }

        return QImage();
    }

}