# src/flow/CMakeLists.txt

file(GLOB FLOW_SRCS "*.cpp")

# Explicitly list node files to ensure they are picked up
list(APPEND FLOW_SRCS
        nodes/source_nodes.cpp
        nodes/infer_nodes.cpp
        nodes/sink_nodes.cpp
)

add_library(xinfer_flow OBJECT ${FLOW_SRCS})

# ... (Rest of dependencies remains the same) ...
target_link_libraries(xinfer_flow PRIVATE
        xinfer_core
        xinfer_zoo
        ${OpenCV_LIBS}
)