#include <gtest/gtest.h>
#include <vector>
#include "abs_npu.h"

class AbsNpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "abs_npu test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "abs_npu test TearDown" << std::endl;
    }
};

TEST_F(AbsNpuTest, abs_npu_test_case_1) {
    // [TODO] define your op here
     ge::op::AbsNpu abs_npu_op;
     ge::TensorDesc tensorDesc;
     ge::Shape shape({32,});
     tensorDesc.SetDataType(ge::DT_FLOAT16);
     tensorDesc.SetShape(shape);

    // [TODO] update op input here
     abs_npu_op.UpdateInputDesc("x1", tensorDesc);

    // [TODO] call InferShapeAndType function here
     auto ret = abs_npu_op.InferShapeAndType();
     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
     auto output_desc = abs_npu_op.GetOutputDesc("y");
     EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
     std::vector<int64_t> expected_output_shape = {32,};
     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
