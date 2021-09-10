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
import os
import csv
import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto


def generate_onnx_model_for_thresholded_relu(input_shape,
                                             output_shape,
                                             data_type,
                                             model_save_path,
                                             alpha=1.0):
    """
    generate onnx model for thresholeded_relu_npu.

    Parameters
    ----------
    input_shape : list
    output_shape : list
    data_type : str, ('float16', 'float32')
    model_save_path: str
    alpha : float, default=1.0

    Returns
    ------
    None
    """
    if data_type.lower() == 'float16':
        elem_type = TensorProto.FLOAT16
    elif data_type.lower() == 'float32':
        elem_type = TensorProto.FLOAT
    else:
        print('[ERROR]: no such data type: {}'.format(data_type))
        return

    vinfo_proto_x = helper.make_tensor_value_info('X', elem_type, input_shape)
    vinfo_proto_y = helper.make_tensor_value_info('Y', elem_type, output_shape)
    node_def = helper.make_node(
        op_type='ThresholdedReluNpu',
        inputs=['X'],
        outputs=['Y'],
        alpha=alpha,
    )
    graph_def = helper.make_graph(
        nodes=[node_def],
        name='ThresholdedReluNpu',
        inputs=[vinfo_proto_x],
        outputs=[vinfo_proto_y]
    )
    model_def = helper.make_model(graph_def, producer_name='thresholded_relu')
    model_def.opset_import[0].version = 11
    onnx.save_model(model_def, model_save_path)
    # print('=====The graph in model=====\n{}'.format(model_def.graph))
    print('[SUCCESS]: save model to path: ' + model_save_path)


def get_amount_of_model(alpha_list, range_list, shape_list):
    """
    get amount of the models to be generate.

    Parameters
    ----------
    alpha_list : list, 1-dimensional list
    range_list : list, 2-dimensional list
    shape_list : list, 2-dimensional list

    Returns
    ------
    model_amount : int
    """
    if len(alpha_list) == len(range_list) and len(range_list) == len(shape_list):
        model_amount = len(alpha_list)
        print('MODEL_AMOUNT = ' + str(model_amount))
        return model_amount
    else:
        print('[ERROR]: model_amount no match!')
        print('| length of alpha_list: ' + str(len(alpha_list)))
        print('| length of range_list: ' + str(len(range_list)))
        print('| length of shape_list: ' + str(len(shape_list)))
        return 0


def generate_onnx_models(alpha_list, range_list, shape_list, data_type):
    """
    generate onnx models for thresholded_relu_npu.

    Parameters
    ----------
    alpha_list : list
    range_list : list
    shape_list : list, 2-dimensional list
    data_type: str

    Returns
    ------
    None
    """
    if data_type.lower() == 'float16':
        MODEL_DATA_TYPE = '_fp16_'
    elif data_type.lower() == 'float32':
        MODEL_DATA_TYPE = '_fp32_'
    else:
        print('[ERROR]: no such data type: {}'.format(data_type))
        return

    MODEL_DIRECTORY = './model/'
    MODEL_NAME_PREFIX = 'Thresholded_relu_npu'
    MODEL_NAME_SUFFIX = '.onnx'
    MODEL_NAME = MODEL_NAME_PREFIX + MODEL_DATA_TYPE

    model_amount = get_amount_of_model(alpha_list, range_list, shape_list)

    if not model_amount:
        return

    if not os.path.exists('./model'):
        os.mkdir('./model')

    for idx in range(0, model_amount):
        order = idx + 1
        order_str = str(order)
        while not order // 100:
            order_str = '0' + order_str
            order *= 10
        model_save_path = MODEL_DIRECTORY + MODEL_NAME + order_str + MODEL_NAME_SUFFIX
        generate_onnx_model_for_thresholded_relu(shape_list[idx],
                                                 shape_list[idx],
                                                 data_type,
                                                 model_save_path,
                                                 alpha_list[idx])


def comupte_thresholded_relu_npu(input_data, alpha=1.0):
    """
    compute expect data of the thresholded_relu_npu.

    Parameters
    ----------
    input_data : numpy.ndarray
    alpha : float, default=1.0

    Returns
    ------
    expect_data: numpy.ndarray
    """
    if alpha < 0:
        raise RuntimeError(
            "In comupte_thresholded_relu_npu():\n" +
            "threshold location of activation ALPHA should greater than or equal 0, " +
            "while ALPHA is " + str(alpha)
        )
    expect_data = input_data.copy()
    expect_data[expect_data <= alpha] = 0
    return expect_data


def read_range_list_from_csv(file_path):
    """
    read data range list from csv file.

    Parameters
    ----------
    file_path : str

    Returns
    ------
    data_range_list: list, 2-dimensional list
    """
    csv_reader = csv.reader(open(file_path, encoding='utf-8'))
    data_range_list = list()
    for row in csv_reader:
        data_range = row[0][1:-1].split(',')
        temp_list = list()
        for el in data_range:
            temp_list.append(float(el))
        data_range_list.append(temp_list)
    return data_range_list


def read_shape_list_from_csv(file_path):
    """
    read shape list from csv file.

    Parameters
    ----------
    file_path : str

    Returns
    ------
    shape_list: list, 2-dimensional list
    """
    csv_reader = csv.reader(open(file_path, encoding='utf-8'))
    shape_list = list()
    for row in csv_reader:
        dim = row[0][2:-2].split(',')
        temp_list = list()
        for el in dim:
            temp_list.append(int(el))
        shape_list.append(temp_list)
    return shape_list


def read_alpha_list_from_csv(file_path):
    """
    read alpha list from csv file.

    Parameters
    ----------
    file_path : str

    Returns
    ------
    alpha_list: list, 1-dimensional list
    """
    csv_reader = csv.reader(open(file_path, encoding='utf-8'))
    alpha_list = list()
    for row in csv_reader:
        alpha = float(row[0][1:-1])
        if alpha < 0:
            raise RuntimeError(
                "In read_alpha_list_from_csv():\n" +
                "threshold location of activation ALPHA should greater than or equal 0, " +
                "while ALPHA is " + str(alpha)
            )
        alpha_list.append(alpha)
    return alpha_list


def generate_input_data_for_thresholded_relu(input_shape,
                                             data_range,
                                             data_type):
    """
    generate uniform input_data by input_shape, data_range and data_type.

    Parameters
    ----------
    input_shape: list, 1-dimensional list
    data_range: list, 1-dimensional list
    data_type: str

    Returns
    ------
    input_data: numpy.ndarray, len(input_shape)-dimensional list
    """
    if data_type == "float16":
        elem_type = np.float16
    elif data_type == "float32":
        elem_type = np.float32
    else:
        print('[ERROR]: no such data type: {}'.format(data_type))
        return
    low = data_range[0]
    high = data_range[1]
    input_data = np.random.uniform(low, high, size=input_shape).astype(elem_type)
    return input_data


def generate_input_shape(low, high, dim, amount):
    """
    generate input_shape_list from "discrete uniform" distribution in interval [low, high).

    Parameters
    ----------
    low: int or list(int), lowest integers to be drawn from the distribution.
    high: int or list(int), the upper bound of the "half-open" interval.
    dim: int, dimension of single input_shape you want to create.
    amount: int, the amount of input_shape.

    Returns
    ------
    input_shape_list: list, 2-dimensional list, the shape of input_shape_list is (amount by dim)
    """
    le_zero_bool = np.less_equal(low, 0)
    le_zero_bool = le_zero_bool.astype(np.int8)
    cnt = np.sum(le_zero_bool)
    if cnt > 0:
        print("[ERROR]: invalid lower bound of interval, "
              "element of the shape should be positive")

    if dim <= 0 or amount <= 0:
        print("[ERROR]: invalid parameters DIM or AMOUNT")
        return
    if dim > 4:
        print("[WARNING]: dim(dimension) > 4")
    size = tuple([amount, dim])
    input_shape_list = np.random.randint(low=low, high=high, size=size)
    input_shape_list = input_shape_list.tolist()
    return input_shape_list


def write_shape_list_to_csv(file_path, shape_list):
    """
    write shape_list to a csv file.

    Parameters
    ----------
    file_path : str
    shape_list: list, 2-dimensional list

    Returns
    ------
    None
    """
    temp_shape_list = list()
    for shape in shape_list:
        temp_shape_list.append('[' + str(shape) + ']')

    with open(file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for temp_shape in temp_shape_list:
            csv_writer.writerow([temp_shape])
    print('[SUCCESS]: write shape_list to csv file: ' + file_path)


def write_input_and_expect_data_to_files(alpha_list, range_list, shape_list, data_type):
    """
    write input data and expect data(ground truth) to files.

    Parameters
    ----------
    alpha_list : list, 1-dimensional list
    range_list : list, 2-dimensional list
    shape_list : list, 2-dimensional list
    data_type : str
    Returns
    ------
    None
    """
    GROUND_TRUTH_PATH = './ground_truth/'
    INPUT_DATA_PATH = './input_data/'
    if not os.path.exists(GROUND_TRUTH_PATH):
        os.mkdir(GROUND_TRUTH_PATH)

    if not os.path.exists(INPUT_DATA_PATH):
        os.mkdir(INPUT_DATA_PATH)

    model_amount = get_amount_of_model(alpha_list, range_list, shape_list)

    if not model_amount:
        return

    for idx in range(model_amount):
        if data_type.lower() == 'float16':
            DATA_TYPE = 'fp16'
        elif data_type.lower() == 'float32':
            DATA_TYPE = 'fp32'
        else:
            print('[ERROR]: no such data type: {}'.format(data_type))
            raise RuntimeError(
                '[ERROR]: no such data type: {}'.format(data_type)
            )

        order = idx + 1
        order_str = str(order)
        while not order // 100:
            order_str = '0' + order_str
            order *= 10

        if not os.path.exists(INPUT_DATA_PATH+'data_{order}'.format(order=order_str)):
            os.mkdir(INPUT_DATA_PATH+'data_{order}'.format(order=order_str))

        input_data_path = INPUT_DATA_PATH+"data_{order}/Thresholded_relu_npu_{data_type}_{order}_in.bin"\
            .format(order=order_str, data_type=DATA_TYPE)
        gt_save_path = GROUND_TRUTH_PATH+"Thresholded_relu_npu_{data_type}_{order}_gt.bin" \
            .format(order=order_str, data_type=DATA_TYPE)

        input_data = generate_input_data_for_thresholded_relu(shape_list[idx], range_list[idx], data_type)
        expect_data = comupte_thresholded_relu_npu(input_data, alpha_list[idx])
        input_data.tofile(input_data_path)
        expect_data.tofile(gt_save_path)
