#include <gtest/gtest.h>
#include <xinfer/xinfer.h>
#include <xinfer/xinfer.h>
#include <vector>

class NmsTest : public ::testing::Test {};

TEST_F(NmsTest, SuppressesOverlappingBoxes) {
    // 1. Create dummy boxes and scores on the CPU
    // Two boxes that highly overlap, one that is separate.
    std::vector<float> h_boxes = {
        10, 10, 50, 50,  // Box 0 (High score)
        12, 12, 52, 52,  // Box 1 (Overlaps Box 0, lower score)
        100, 100, 150, 150 // Box 2 (Separate)
    };
    std::vector<float> h_scores = { 0.9f, 0.8f, 0.85f };

    // 2. Upload to GPU tensors
    xinfer::core::Tensor d_boxes({3, 4}, xinfer::core::DataType::kFLOAT);
    xinfer::core::Tensor d_scores({3}, xinfer::core::DataType::kFLOAT);
    d_boxes.copy_from_host(h_boxes.data());
    d_scores.copy_from_host(h_scores.data());

    // 3. Run the NMS function
    float iou_threshold = 0.5f;
    std::vector<int> kept_indices = xinfer::postproc::detection::nms(d_boxes, d_scores, iou_threshold);

    // 4. Verify the results
    // We expect Box 1 to be suppressed by Box 0.
    // Box 0 and Box 2 should be kept.
    ASSERT_EQ(kept_indices.size(), 2);

    // Sort for consistent checking
    std::sort(kept_indices.begin(), kept_indices.end());
    ASSERT_EQ(kept_indices[0], 0);
    ASSERT_EQ(kept_indices[1], 2);
}