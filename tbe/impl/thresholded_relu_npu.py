#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
thresholded_relu_npu
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic

# pylint: disable=invalid-name,unused-argument
@fusion_manager.register("thresholded_relu_npu")
def thresholded_relu_npu_compute(x, y, alpha=1.0, kernel_name="thresholded_relu_npu"):
    """
    algorithm:
       f(x)= x, when x>alpha
       f(x)= 0, otherwise

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    y: dict
        dict info of y
    alpha : float
        default 1.0
    kernel_name: str
        kernel name, default value is "thresholded_relu_npu"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    input_data_type = x.dtype.lower()
    if alpha == 0.0:
        data_res = te.lang.cce.vrelu(x)
        res = te.lang.cce.cast_to(data_res, input_data_type)
    else:
        scalar_zero = tvm.const(0, dtype=input_data_type)
        scalar_alpha = tvm.const(alpha, dtype=input_data_type)
        res = te.lang.cce.vcmpsel(x, scalar_alpha, 'ge', x, scalar_zero)
    return res
    # shape_input = x.shape
    # # alpha == 0
    # if alpha == 0.0:
    #     if input_data_type in "float32":
    #         tensor_zero = te.lang.cce.broadcast(tvm.const(0, input_data_type), shape_input)
    #         data_res = te.lang.cce.vmax(x, tensor_zero)
    #     else:
    #         # input_data_type in "float16"
    #         data_res = te.lang.cce.vrelu(x)
    #     data_res = te.lang.cce.cast_to(data_res, input_data_type)
    # # alpha > 0
    # else:
    #     scalar_zero = tvm.const(0, input_data_type)
    #     scalar_alpha = tvm.const(alpha, input_data_type)
    #     data_res = te.lang.cce.vcmpsel(x, scalar_alpha, 'ge', x, scalar_zero)
    # return data_res


def thresholded_relu_npu(x, y, alpha=1.0, kernel_name="thresholded_relu_npu"):
    """
    calculate thresholded_relu

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16, float32.
    y : dict
        shape and dtype of input, should be the same shape and dtype as input.
    alpha : float
        default 1.0
    kernel_name : str
        cce kernel name, default value is "thresholded_relu_npu"

    Returns
    ------
    None
    """
    # check input tensor shape
    shape_x = x.get("shape")
    input_data_type = x.get("dtype").lower()

    # check input tensor data type
    check_list = ["float16", "float32"]
    if input_data_type not in check_list:
        raise RuntimeError(
            "thresholded relu only support %s while dtype is %s"
            % (",".join(check_list), input_data_type))

    # check param: alpha
    if alpha < 0:
        raise RuntimeError(
            "threshold location of activation ALPHA should greater than or equal 0."
        )

    # set placeholder for input
    input_data_x = tvm.placeholder(shape_x, name="input_data_x", dtype=input_data_type)

    with tvm.target.cce():
        # call _compute function
        result = thresholded_relu_npu_compute(input_data_x, y, alpha, kernel_name)
        # auto schedule
        schedule = generic.auto_schedule(result)

    config = {
        "print_ir": True,
        "name": kernel_name,
        "tensor_list": [input_data_x, result]
    }
    # build
    te.lang.cce.cce_build_code(schedule, config)
