#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
thresholded_relu_npu
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils import para_check

EPSINON = 1e-6

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
    if alpha - 0.0 < EPSINON:
        data_res = te.lang.cce.vrelu(x)
        res = te.lang.cce.cast_to(data_res, input_data_type)
    else:
        scalar_zero = tvm.const(0, dtype=input_data_type)
        scalar_alpha = tvm.const(alpha, dtype=input_data_type)
        res = te.lang.cce.vcmpsel(x, scalar_alpha, 'gt', x, scalar_zero)
    return res


# pylint: disable=redefined-builtin
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
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
    para_check.check_shape(shape=shape_x, param_name='x')

    # check input tensor data type
    check_list = ("float16", "float32")
    input_dtype = x.get("dtype").lower()
    para_check.check_dtype(dtype=input_dtype, check_list=check_list, param_name='x')

    # check param: alpha(float) should >= 0
    _check_param_alpha(alpha=alpha, op_name='thresholded_relu_npu')

    # set placeholder for input
    input_data_x = tvm.placeholder(shape_x, dtype=input_dtype, name="input_data_x")

    with tvm.target.cce():
        result = thresholded_relu_npu_compute(input_data_x, y, alpha, kernel_name)
        schedule = generic.auto_schedule(result)

    config = {
        "print_ir": True,
        "name": kernel_name,
        "tensor_list": [input_data_x, result]
    }
    # build
    te.lang.cce.cce_build_code(schedule, config)


def _check_param_alpha(alpha, op_name):
    if alpha < 0:
        error_info = {
            'op_name': op_name,
            'attr_name': 'alpha',
            'attr_actual_value': alpha
        }
        raise RuntimeError(
            "In op[%s], the threshold location of activation [%s] should greater than or equal 0, "
            "but actually is [%s]." % (error_info['op_name'],
                                       error_info['attr_name'],
                                       error_info['attr_actual_value']))
