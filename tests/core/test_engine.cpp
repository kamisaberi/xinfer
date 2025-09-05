#include <gtest/gtest.h>
#include <xinfer/xinfer.h>
#include <xinfer/xinfer.h>
#include <fstream>

class EngineTest : public ::testing::Test {
protected:
    // This test assumes a simple pre-built engine exists.
    // For example, an ONNX model with one input [1,3,224,224] and one output [1,1000].
    const std::string engine_path = "assets/dummy_classifier.engine";

    void SetUp() override {
        if (!std::ifstream(engine_path).good()) {
            FAIL() << "Test engine file not found: " << engine_path
                   << "\nPlease create a dummy engine before running this test.";
        }
    }
};

TEST_F(EngineTest, ConstructorLoadsEngine) {
    ASSERT_NO_THROW({
        xinfer::core::InferenceEngine engine(engine_path);
    });
}

TEST_F(EngineTest, ThrowsOnInvalidPath) {
    ASSERT_THROW(xinfer::core::InferenceEngine engine("invalid/path/does/not/exist.engine"), std::runtime_error);
}

TEST_F(EngineTest, IntrospectionIsCorrect) {
    xinfer::core::InferenceEngine engine(engine_path);

    ASSERT_EQ(engine.get_num_inputs(), 1);
    ASSERT_EQ(engine.get_num_outputs(), 1);

    // Note: TensorRT may optimize the batch dimension to -1 (dynamic).
    // A robust test would check the other dimensions.
    auto input_shape = engine.get_input_shape(0);
    ASSERT_EQ(input_shape.size(), 4); // B, C, H, W
    ASSERT_EQ(input_shape[1], 3);
    ASSERT_EQ(input_shape[2], 224);
    ASSERT_EQ(input_shape[3], 224);

    auto output_shape = engine.get_output_shape(0);
    ASSERT_EQ(output_shape.size(), 2); // B, NumClasses
    ASSERT_EQ(output_shape[1], 1000);
}

TEST_F(EngineTest, SynchronousInference) {
    xinfer::core::InferenceEngine engine(engine_path);

    auto input_shape = engine.get_input_shape(0);
    input_shape[0] = 1; // Set a specific batch size
    xinfer::core::Tensor input_tensor(input_shape, xinfer::core::DataType::kFLOAT);

    std::vector<xinfer::core::Tensor> output_tensors;
    ASSERT_NO_THROW({
        output_tensors = engine.infer({input_tensor});
    });

    ASSERT_EQ(output_tensors.size(), 1);
    const auto& output_tensor = output_tensors[0];

    auto expected_output_shape = engine.get_output_shape(0);
    expected_output_shape[0] = 1; // Batch size of 1

    ASSERT_EQ(output_tensor.shape(), expected_output_shape);
    ASSERT_NE(output_tensor.data(), nullptr);
}