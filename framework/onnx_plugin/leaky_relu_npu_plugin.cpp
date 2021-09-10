/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: plugin for leakyrelu onnx operator
 * Author:
 * Create: 2020-11-20
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "ge_onnx.pb.h"
#include "register/register.h"
#include <string>

namespace domi {
Status ParseParamsLeakyRelu(const Message *op_src, ge::Operator& op_dst)
{
    const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        // OP_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    float negativeSlope = 0.0f;
    bool bFindAlpha = false;
    for (auto attr : node->attribute()) {
        if (attr.name() == "alpha") {
            bFindAlpha = true;
            negativeSlope = attr.f();
            break;
        }
    }

    if (!bFindAlpha) {
        negativeSlope = 0.01f;
    }
    op_dst.SetAttr("negative_slope", negativeSlope);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyReluNpu")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::LeakyReluNpu")
    .ParseParamsFn(ParseParamsLeakyRelu)
    .ImplyType(ImplyType::TVM);
}