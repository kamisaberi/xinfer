#include <gtest/gtest.hh>
#include <xinfer/core/tensor.h>
#include <vector>
#include <numeric>

// A test fixture can be used to set up common objects for multiple tests.
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // This is run before each test in this suite.
    }
    // You can also add a TearDown() method.
};

// TEST_F uses the fixture (TensorTest)
TEST_F(TensorTest, ConstructorAndAllocation) {
    std::vector<int64_t> shape = {1, 3, 224, 224};
    xinfer::core::Tensor tensor(shape, xinfer::core::DataType::kFLOAT);

    ASSERT_NE(tensor.data(), nullptr);
    ASSERT_EQ(tensor.shape(), shape);
    ASSERT_EQ(tensor.dtype(), xinfer::core::DataType::kFLOAT);

    size_t expected_elements = 1 * 3 * 224 * 224;
    ASSERT_EQ(tensor.num_elements(), expected_elements);
    ASSERT_EQ(tensor.size_bytes(), expected_elements * sizeof(float));
}

TEST_F(TensorTest, MoveSemantics) {
    std::vector<int64_t> shape = {1, 10};
    xinfer::core::Tensor tensor1(shape, xinfer::core::DataType::kFLOAT);
    void* original_ptr = tensor1.data();

    ASSERT_NE(original_ptr, nullptr);

    // Test move constructor
    xinfer::core::Tensor tensor2 = std::move(tensor1);

    // The original tensor should now be empty (null)
    ASSERT_EQ(tensor1.data(), nullptr);
    ASSERT_EQ(tensor1.num_elements(), 0);

    // The new tensor should have the original data
    ASSERT_EQ(tensor2.data(), original_ptr);
    ASSERT_EQ(tensor2.shape(), shape);

    // Test move assignment
    xinfer::core::Tensor tensor3;
    tensor3 = std::move(tensor2);

    ASSERT_EQ(tensor2.data(), nullptr);
    ASSERT_EQ(tensor3.data(), original_ptr);
}

TEST_F(TensorTest, HostDeviceCopy) {
    std::vector<int64_t> shape = {1, 1024};
    xinfer::core::Tensor tensor(shape, xinfer::core::DataType::kFLOAT);

    // Create some data on the CPU
    std::vector<float> h_input(1024);
    std::iota(h_input.begin(), h_input.end(), 0.0f); // Fill with 0, 1, 2, ...

    // Copy from host to device
    tensor.copy_from_host(h_input.data());

    // Create a new host buffer to copy back to
    std::vector<float> h_output(1024, -1.0f); // Fill with a different value

    // Copy from device to host
    tensor.copy_to_host(h_output.data());

    // Verify that the data is identical
    for (size_t i = 0; i < h_input.size(); ++i) {
        ASSERT_FLOAT_EQ(h_input[i], h_output[i]);
    }
}

TEST_F(TensorTest, ThrowsOnInvalidUse) {
    xinfer::core::Tensor empty_tensor;
    std::vector<float> h_buffer(10);

    // Expect an exception when trying to use an unallocated tensor
    ASSERT_THROW(empty_tensor.copy_to_host(h_buffer.data()), std::runtime_error);
}