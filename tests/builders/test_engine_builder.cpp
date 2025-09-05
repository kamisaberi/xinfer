#include <gtest/gtest.h>
#include <xinfer/builders/engine_builder.h>
#include <fstream>

class EngineBuilderTest : public ::testing::Test {
protected:
    // This test requires a simple, valid ONNX model file.
    const std::string onnx_path = "assets/dummy_model.onnx";
    const std::string output_engine_path = "test_builder_output.engine";

    void SetUp() override {
        if (!std::ifstream(onnx_path).good()) {
            FAIL() << "Test ONNX file not found: " << onnx_path
                   << "\nPlease create a dummy ONNX model before running this test.";
        }
    }

    void TearDown() override {
        // Clean up the generated engine file after each test
        remove(output_engine_path.c_str());
    }
};

TEST_F(EngineBuilderTest, BuildsFp32EngineSuccessfully) {
    xinfer::builders::EngineBuilder builder;

    ASSERT_NO_THROW({
        builder.from_onnx(onnx_path)
               .with_max_batch_size(1)
               .build_and_save(output_engine_path);
    });

    // Verify that the output file was actually created
    ASSERT_TRUE(std::ifstream(output_engine_path).good());
}

TEST_F(EngineBuilderTest, BuildsFp16EngineSuccessfully) {
    xinfer::builders::EngineBuilder builder;

    ASSERT_NO_THROW({
        builder.from_onnx(onnx_path)
               .with_fp16()
               .with_max_batch_size(4)
               .build_and_save(output_engine_path);
    });

    ASSERT_TRUE(std::ifstream(output_engine_path).good());
}

TEST_F(EngineBuilderTest, ThrowsOnInvalidOnnxPath) {
    xinfer::builders::EngineBuilder builder;
    builder.from_onnx("invalid/path/does/not/exist.onnx");

    ASSERT_THROW(builder.build_and_save(output_engine_path), std::runtime_error);
}