#include "results_panel.h"
#include <QVBoxLayout>
#include <QHeaderView>

ResultsPanel::ResultsPanel(QWidget *parent) : QWidget(parent) {
    setupUI();
}

void ResultsPanel::setupUI() {
    auto layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    m_tree = new QTreeWidget(this);
    m_tree->setColumnCount(3);
    m_tree->setHeaderLabels({"ID/Label", "Confidence", "Position"});
    m_tree->header()->setSectionResizeMode(0, QHeaderView::Stretch);

    // Style for Dark Theme
    m_tree->setStyleSheet(
        "QTreeWidget { background-color: #1e1e1e; border: 1px solid #444; }"
        "QHeaderView::section { background-color: #333; color: white; border: none; padding: 4px; }"
    );

    layout->addWidget(m_tree);
}

void ResultsPanel::clear() {
    m_tree->clear();
}

void ResultsPanel::updateDetections(const std::vector<xinfer::postproc::BoundingBox>& boxes) {
    m_tree->clear();

    int id_counter = 0;
    for (const auto& box : boxes) {
        auto item = new QTreeWidgetItem(m_tree);

        // Col 0: Label/ID
        item->setText(0, "Class " + QString::number(box.class_id));

        // Col 1: Score
        item->setText(1, QString::number(box.confidence * 100, 'f', 1) + "%");

        // Col 2: Position
        QString pos = QString("[%1, %2]").arg((int)box.x1).arg((int)box.y1);
        item->setText(2, pos);

        id_counter++;
    }
}