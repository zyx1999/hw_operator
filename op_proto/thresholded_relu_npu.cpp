/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file thresholded_relu_npu.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "thresholded_relu_npu.h"
#include <vector>
#include <string>

namespace ge{
    IMPLEMT_INFERFUNC(ThresholdedReluNpu, ThresholdedReluNpuInferShape){
        vector<int64_t> x_shape = op.GetInputDesc("x").GetShape().GetDims();
        DataType x_dtype = op.GetInputDesc("x").GetDataType();
        TensorDesc y_desc = op.GetOutputDesc("y");
        y_desc.SetShape(ge::Shape(x_shape));
        y_desc.SetDataType(x_dtype);
        (void)op.UpdateOutputDesc("y", y_desc);
        return GRAPH_SUCCESS;
    }
    IMPLEMT_VERIFIER(ThresholdedReluNpu, ThresholdedReluNpuVerify){
        return GRAPH_SUCCESS;
    }
    INFER_FUNC_REG(ThresholdedReluNpu, ThresholdedReluNpuInferShape);
    VERIFY_FUNC_REG(ThresholdedReluNpu, ThresholdedReluNpuVerify);
}