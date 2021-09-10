#include <gtest/gtest.h>
#include <vector>
#include "leaky_relu_npu.h"

class LeakyReluNpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "leaky_relu_npu test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "leaky_relu_npu test TearDown" << std::endl;
    }
};

TEST_F(LeakyReluNpuTest, leaky_relu_npu_test_case_1) {
    // [TODO] define your op here
     ge::op::LeakyReluNpu leaky_relu_npu_op;
     ge::TensorDesc tensorDesc;
     ge::Shape shape({1,16, 128, 128});
     tensorDesc.SetDataType(ge::DT_FLOAT16);
     tensorDesc.SetShape(shape);

    // [TODO] update op input here
     leaky_relu_npu_op.UpdateInputDesc("x1", tensorDesc);

    // [TODO] call InferShapeAndType function here
     auto ret = leaky_relu_npu_op.InferShapeAndType();
     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
     auto output_desc = leaky_relu_npu_op.GetOutputDesc("y");
     EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
     std::vector<int64_t> expected_output_shape = {1,16, 128, 128};
     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
