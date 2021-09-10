/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "ge_onnx.pb.h"
#include "register/register.h"
#include <string>
#include <vector>

namespace domi {
/*
用户自定义并实现的类函数，完成模型参数和权值的转换，将记过填到operator类中
op_src  表示输入   protobuf格式的数据结构（来源于caffe模型的prototxt文件），包含算子参数信息
op_dest 表示输出   davinci离线模型的算子数据结构，保存算子信息。
关于 operator类，可以参考《Ascend 310 GE API 参考》中的“operator”类接口
*/
Status ParseParamsAbs(const Message *op_src, ge::Operator &op_dest)   
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        printf("Dynamic cast Abs to NodeProto  failed\n");
        return FAILED;
    }
    return SUCCESS;
}

// register Abs op info to GE
REGISTER_CUSTOM_OP("AbsNpu")  //设置算子的注册名称
    .FrameworkType(ONNX)   //设置算子原始框架类型
    .OriginOpType("ai.onnx::11::AbsNpu")    //原始框架的算子名称
    .ParseParamsFn(ParseParamsAbs)     //注册解析算子参数的回调函数,在tf中一般不用自己编写，主要防止原始框架的参数和在TBE中的运行的参数不同导致报错
    .ImplyType(ImplyType::TVM);
}  // namespace domi
