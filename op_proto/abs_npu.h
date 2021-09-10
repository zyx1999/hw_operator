/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: op_proto for srelu operator
 * Author:
 * Create: 2020-11-21
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 * file s_relu.h
 */

#ifndef GE_OP_ABSNPU_H
#define GE_OP_ABSNPU_H

#include "graph/operator_reg.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
/**
* @brief piecewise linear activation function
*
* @par Inputs:
*  5 inputs, including:
* @li x : input data. required
**/
REG_OP(AbsNpu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(AbsNpu)
}
#endif // GE_OP_ABSNPU_H
