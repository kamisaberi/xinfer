#pragma once

#include <QObject>
#include <QString>
#include <QVariantMap>

// xInfer Compiler Headers
#include <xinfer/compiler/base_compiler.h>

class CompilerController : public QObject {
    Q_OBJECT
public:
    explicit CompilerController(QObject *parent = nullptr);

public slots:
    /**
     * @brief Start the compilation process.
     *
     * @param targetName The target platform string (e.g., "nv-trt", "rockchip-rknn").
     * @param inputPath Path to the .onnx or .tflite file.
     * @param outputPath Desired output path.
     * @param precision "fp32", "fp16", or "int8".
     * @param extraParams Additional flags (e.g. calibration data path).
     */
    void compileModel(QString targetName, QString inputPath, QString outputPath, QString precision, QVariantMap extraParams);

    signals:
        void logOutput(QString message);
    void progressChanged(int percent); // 0-100
    void compilationFinished(bool success, QString message);

private:
    // Helper to map QString to xInfer Enums
    xinfer::compiler::Target mapTarget(const QString& t);
    xinfer::compiler::Precision mapPrecision(const QString& p);
};