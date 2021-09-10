/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: op_proto for srelu operator
Author:
Create: 2020-11-21
*/
#include "abs_npu.h"

namespace ge {
    IMPLEMT_VERIFIER(AbsNpu, AbsNpuVerify)
    {
        return GRAPH_SUCCESS;
    }
    IMPLEMT_COMMON_INFERFUNC(AbsNpuInferShape){
        vector<int64_t> x_shape = op.GetInputDesc("x").GetShape().GetDims();
        TensorDesc td = op.GetOutputDesc("y");
        td.SetShape(ge::Shape(x_shape));
        DataType input_dtype = op.GetInputDesc("x").GetDataType();
        td.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
    }
    COMMON_INFER_FUNC_REG(AbsNpu, AbsNpuInferShape);
    VERIFY_FUNC_REG(AbsNpu, AbsNpuVerify);
}
