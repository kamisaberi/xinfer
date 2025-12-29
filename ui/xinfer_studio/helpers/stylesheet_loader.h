#pragma once

#include <QString>

// Forward declaration to avoid heavy includes in the header
class QApplication;

namespace xinfer::ui::helpers {

    /**
     * @brief Loads a QSS file and applies it to the application.
     *
     * @param app Reference to the main QApplication instance.
     * @param resourcePath Path to the .qss file (e.g., ":/styles/dark_theme.qss").
     */
    void apply_theme(QApplication& app, const QString& resourcePath);

} // namespace xinfer::ui::helpers