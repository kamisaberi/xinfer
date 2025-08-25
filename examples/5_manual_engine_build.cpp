#include <include/builders/engine_builder.h>
#include <iostream>

int main() {
    std::string onnx_path = "assets/my_custom_model.onnx";
    std::string engine_path = "my_custom_model_fp16.engine";

    std::cout << "Building custom TensorRT engine from: " << onnx_path << std::endl;

    try {
        xinfer::builders::EngineBuilder builder;
        builder.from_onnx(onnx_path)
               .with_fp16()
               .with_max_batch_size(16);

        builder.build_and_save(engine_path);

        std::cout << "Successfully built engine and saved to: " << engine_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during engine build: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}