cmake_minimum_required(VERSION 3.20)
project(xInfer_Examples)

find_package(OpenCV REQUIRED)

# Link against your library
link_directories(${CMAKE_SOURCE_DIR}/../build) # Point to libxinfer.so

# Macro to add example
macro(add_xinfer_example name path)
    add_executable(${name} ${path})
    target_link_libraries(${name} PRIVATE
            xinfer_core xinfer_preproc xinfer_postproc xinfer_backends
            ${OpenCV_LIBS}
    )
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/../include)
endmacro()

# Define Targets
add_xinfer_example(demo_detection   01_computer_vision/object_detection.cpp)
add_xinfer_example(demo_siem        04_tabular_siem/network_intrusion.cpp)
add_xinfer_example(demo_aegis       06_projects/aegis_sky/main.cpp)
add_xinfer_example(demo_chat        03_nlp_text/llama_chatbot.cpp)