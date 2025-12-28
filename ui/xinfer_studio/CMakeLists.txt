cmake_minimum_required(VERSION 3.20)
project(xInfer_Studio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON) # Qt Magic: Auto Meta-Object Compiler
set(CMAKE_AUTOUIC ON) # Qt Magic: Auto UI Compiler
set(CMAKE_AUTORCC ON) # Qt Magic: Auto Resource Compiler

# --- Find Dependencies ---
find_package(Qt6 COMPONENTS Widgets Core Gui Network REQUIRED)
find_package(OpenCV REQUIRED)

# --- Find xInfer Libraries ---
# Assuming xInfer root build is available
link_directories(${CMAKE_SOURCE_DIR}/../../build)
include_directories(${CMAKE_SOURCE_DIR}/../../include)

# --- Define Source Files ---
set(UI_SOURCES
        main.cpp
        resources.qrc # Add this line

        # App
        app/mainwindow.cpp
        app/mainwindow.h
        app/mainwindow.ui

        # Core
        core/device_manager.cpp
        core/device_manager.h

        # Controllers (Threading)
        controllers/zoo_controller.cpp
        controllers/zoo_controller.h

        # Widgets (Custom UI)
        widgets/video_display.cpp
        widgets/video_display.h

        # Helpers
        helpers/qt_cv_converter.cpp
        helpers/qt_cv_converter.h
)

# --- Add Executable ---
add_executable(xinfer_studio ${UI_SOURCES})

# --- Link Libraries ---
target_link_libraries(xinfer_studio PRIVATE
        Qt6::Widgets
        Qt6::Core
        Qt6::Gui
        Qt6::Network
        ${OpenCV_LIBS}

        # Link against your backend
        xinfer_zoo
        xinfer_core
        xinfer_backends
        xinfer_preproc
        xinfer_postproc
)