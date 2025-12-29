#pragma once

#include <QWidget>
#include <vector>
#include <xinfer/postproc/vision/types.h>

namespace Ui {
    class ResultsPanel;
}

class ResultsPanel : public QWidget {
    Q_OBJECT

public:
    explicit ResultsPanel(QWidget *parent = nullptr);
    ~ResultsPanel();

    void clear();
    void updateDetections(const std::vector<xinfer::postproc::BoundingBox>& boxes);

private:
    Ui::ResultsPanel *ui;
};