# # -*- coding:utf-8 -*-
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = BroadcastOpUT(op_type ="abs_npu" , op_module_name = "impl.abs_npu", op_func_name = "abs")

ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (32, ),
        "ori_shape": (32,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32"
    },  {
        "shape": (32,),
        "ori_shape": (32,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32"
    }]
})
ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (32, ),
        "ori_shape": (32,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int32"
    },  {
        "shape": (32,),
        "ori_shape": (32,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int32"
    }]
})

def np_absnpu(x, y):
    '''input tensor(dict): x
       output tensor(dict): y
    '''
    y=np.abs(x.get("value"))
    return y

ut_case.add_precision_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (32, ),
        "ori_shape": (32,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "input",
        "value_range": [-10.0, 10.0]

},  {
        "shape": (32,),
        "ori_shape": (32,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "output"
    }],

    "case_name": "test_abs_npu_precision",
    "calc_expect_func": np_absnpu,
})