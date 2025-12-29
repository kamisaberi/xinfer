#pragma once

#include <QWidget>
#include <QTreeWidget>
#include <vector>

// Include types to display
#include <xinfer/postproc/vision/types.h>

class ResultsPanel : public QWidget {
    Q_OBJECT
public:
    explicit ResultsPanel(QWidget *parent = nullptr);

    // Clear the table
    void clear();

    // Update with new detections
    void updateDetections(const std::vector<xinfer::postproc::BoundingBox>& boxes);

    // Update with classification results
    // void updateClassifications(...)

private:
    QTreeWidget* m_tree;
    void setupUI();
};