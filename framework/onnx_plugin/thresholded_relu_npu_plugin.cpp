/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: plugin for thresholdedrelu onnx operator
 * Author: zhaoyuxuan
 * Create: 2021-09-01
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
    Status ParseParamsThresholdedRelu(const Message *op_src, ge::Operator& op_dst)
    {
        const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
        if (node == nullptr) {
            // OP_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
            return FAILED;
        }

        float threshold = 1.0f;
        bool bFindAlpha = false;
        for (auto attr : node->attribute()) {
            if (attr.name() == "alpha") {
                bFindAlpha = true;
                threshold = attr.f();
                break;
            }
        }

        if (!bFindAlpha) {
            threshold = 1.0f;
        }
        op_dst.SetAttr("threshold", threshold);
        return SUCCESS;
    }

    REGISTER_CUSTOM_OP("ThresholdedReluNpu")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::ThresholdedReluNpu")
    .ParseParamsFn(ParseParamsThresholdedRelu)
    .ImplyType(ImplyType::TVM);
//    Status ParseParamsThresholdedRelu(const Message *op_src, ge::Operator& op_dst)
//    {
//        const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
//        if (node == nullptr) {
//            //OP_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
//            printf("Dynamic cast op_src to NodeProto failed.");
//            return FAILED;
//        }
//
//        float alpha = 0.0f;
//        bool bFindAlpha = false;
//        for (auto attr : node->attribute()) {
//            if (attr.name() == "alpha") {
//                bFindAlpha = true;
//                alpha = attr.f();
//                break;
//            }
//        }
//
//        if (!bFindAlpha) {
//            alpha = 1.0f;
//        }
//        op_dst.SetAttr("alpha", alpha);
//        return SUCCESS;
//    }
//    // register thresholded relu op info to GE
//    REGISTER_CUSTOM_OP("ThresholdedReluNpu")
//    .FrameworkType(ONNX)
//    .OriginOpType("ai.onnx::11::ThresholdedReluNpu")
//    .ParseParamsFn(ParseParamsThresholdedRelu)
//    .ImplyType(ImplyType::TVM);
} // namespace domi