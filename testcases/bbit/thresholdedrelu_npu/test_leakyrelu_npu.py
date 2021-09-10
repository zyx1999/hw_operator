"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""
import numpy as np
import onnx

def generate_model(input_shape, alpha, output_shape, model_save_path):

    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)    
    
    attr = {"alpha": alpha}

    node_def = onnx.helper.make_node(
    op_type='LeakyReluNpu',
    inputs=['X'],
    outputs=['Y'],
    alpha = alpha
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'LeakyReluNpu',
        [X],
        [Y],
    )
    model_def = onnx.helper.make_model(graph_def,
                              producer_name='leakey_relu')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, model_save_path)

def comupte_leakey_relu_npu(input_shape, alpha, input_data_path, gt_save_path):
    
    input_data = np.random.random(input_shape).astype(np.float32)*10 - 5
    
    expect_data = input_data.copy()
    expect_data[expect_data < 0] = expect_data[expect_data < 0] * alpha

    return input_data, expect_data


if __name__ == "__main__":

    input_shape = [1,16,128,128]
    output_shape = [1,16,128,128]
    alpha = 0.1

    model_save_path = "./model/leakyrelu_{}_{}_{}_{}.onnx".format(*input_shape)
    input_data_path = "./input_data/leakyrelu_in_{}_{}_{}_{}.bin".format(*input_shape)
    gt_save_path = "./ground_truth/leakyrelu_gt_{}_{}_{}_{}.bin".format(*output_shape)

    generate_model(input_shape, alpha, output_shape, model_save_path)
    input_data, expect_data = comupte_leakey_relu_npu(input_shape, alpha, input_data_path, gt_save_path)

    input_data.tofile(input_data_path)
    expect_data.tofile(gt_save_path)
    
    print("Done")
