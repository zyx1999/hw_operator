#include <gtest/gtest.h>
#include <vector>
#include "thresholded_relu_npu.h"

class ThresholdedReluNpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "thresholded_relu_npu test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "thresholded_relu_npu test TearDown" << std::endl;
    }
};
// infer dtype: DT_FLOAT16
TEST_F(ThresholdedReluNpuTest, thresholded_relu_npu_test_case_1) {
    // [TODO] define your op here
    ge::op::ThresholdedReluNpu thresholded_relu_npu_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({1, 16, 128, 128});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);

    // [TODO] update op input here
    thresholded_relu_npu_op.UpdateInputDesc("x", tensorDesc);

    // [TODO] call InferShapeAndType function here
    auto ret = thresholded_relu_npu_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = thresholded_relu_npu_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 16, 128, 128};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
// infer dtype: DT_FLOAT
TEST_F(ThresholdedReluNpuTest, thresholded_relu_npu_test_case_2) {
    // [TODO] define your op here
    ge::op::ThresholdedReluNpu thresholded_relu_npu_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({1, 16, 128, 128});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);

    // [TODO] update op input here
    thresholded_relu_npu_op.UpdateInputDesc("x", tensorDesc);

    // [TODO] call InferShapeAndType function here
    auto ret = thresholded_relu_npu_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = thresholded_relu_npu_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1, 16, 128, 128};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}