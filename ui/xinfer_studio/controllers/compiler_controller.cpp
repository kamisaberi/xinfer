#include "compiler_controller.h"
#include <xinfer/compiler/compiler_factory.h>
#include <xinfer/core/logging.h>

#include <QDebug>
#include <QFileInfo>
#include <QDir>

using namespace xinfer;

CompilerController::CompilerController(QObject *parent) : QObject(parent) {}

void CompilerController::compileModel(QString targetName, QString inputPath, QString outputPath, QString precision, QVariantMap extraParams) {
    emit progressChanged(0);
    emit logOutput("Initializing Compiler for target: " + targetName);

    compiler::CompileConfig config;

    // 1. Map Inputs
    try {
        config.target = mapTarget(targetName);
        config.precision = mapPrecision(precision);
    } catch (const std::exception& e) {
        emit compilationFinished(false, QString::fromStdString(e.what()));
        return;
    }

    config.input_path = inputPath.toStdString();
    config.output_path = outputPath.toStdString();

    // Map Extra Params
    if (extraParams.contains("calibration_data")) {
        config.calibration_data_path = extraParams["calibration_data"].toString().toStdString();
    }

    // Vendor Params (e.g. CORE=0)
    // Assuming UI passes them as a string list or map
    if (extraParams.contains("vendor_flags")) {
        QStringList flags = extraParams["vendor_flags"].toStringList();
        for(const auto& f : flags) {
            config.vendor_params.push_back(f.toStdString());
        }
    }

    // 2. Create Driver
    auto driver = compiler::CompilerFactory::create(config.target);
    if (!driver) {
        emit compilationFinished(false, "Unknown target driver: " + targetName);
        return;
    }

    emit logOutput("Driver selected: " + QString::fromStdString(driver->get_name()));

    // 3. Validation
    emit progressChanged(10);
    emit logOutput("Validating toolchain environment...");

    if (!driver->validate_environment()) {
        emit compilationFinished(false, "Toolchain missing. Please check xInfer installation.");
        return;
    }

    // 4. Compile (Blocking Call)
    emit progressChanged(20);
    emit logOutput("Starting compilation (this may take a while)...");

    // NOTE: In a real GUI, we might want to capture stdout/stderr from the driver.
    // Since xInfer drivers use std::system, we can't easily capture it here without piping.
    // For now, we rely on the final result.

    bool success = false;
    try {
        success = driver->compile(config);
    } catch (const std::exception& e) {
        emit compilationFinished(false, QString::fromStdString(e.what()));
        return;
    }

    emit progressChanged(100);

    if (success) {
        emit logOutput("Success! Engine saved to: " + outputPath);
        emit compilationFinished(true, "Compilation Successful");
    } else {
        emit logOutput("Compilation failed. Check console/logs for driver errors.");
        emit compilationFinished(false, "Compilation Failed");
    }
}

xinfer::compiler::Target CompilerController::mapTarget(const QString& t) {
    return compiler::stringToTarget(t.toStdString());
}

xinfer::compiler::Precision CompilerController::mapPrecision(const QString& p) {
    return compiler::stringToPrecision(p.toStdString());
}