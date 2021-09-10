# # -*- coding:utf-8 -*-
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT(op_type ="leakey_relu_npu" , op_module_name = "impl.leaky_relu_npu", op_func_name = "leaky_relu_demo")

ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [
        # x
        {"shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32" },
        # y
        {"shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32"},
        #negative_slope
        0,
    ]
})
ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
    },  {
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16"
    },
        0,
    ]
})
ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
    },  {
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16"
    },
        -1,
    ]
})
ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
    },  {
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16"
    },
        2,
    ]
})
ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "int8",
    },  {
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "int8"
    },
        -1,
    ]
})
ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "int8",
    },  {
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "int8"
    },
        2,
    ]
})
ut_case.add_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float64",
    },  {
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float64"
    },
        2,
    ]
})

def np_leaky_relu_npu(x, y, negative_slope):
    """input tensor(dict) : x
       output tensor(dict) : y
    """
    # input_tensor : numpy array
    input_tensor = x.get("value")
    input_tensor[input_tensor < 0] = input_tensor[input_tensor < 0] * negative_slope

    return input_tensor

ut_case.add_precision_case(support_soc=["Ascend610", "Ascend310"], case={
    "params": [{
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "input",
        "value_range": [-10.0, 10.0]
    },  {
        "shape": (1,16,128,128),
        "ori_shape": (1,16,128,128),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "output"
    },
    0.1
    ],
    "case_name": "test_leaky_relu_npu_precision",
    "calc_expect_func": np_leaky_relu_npu,
})