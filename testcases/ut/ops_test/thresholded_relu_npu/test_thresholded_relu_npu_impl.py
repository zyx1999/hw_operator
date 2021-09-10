# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info

ut_case = BroadcastOpUT(op_type="thresholded_relu_npu", op_module_name="impl.thresholded_relu_npu",
                        op_func_name="thresholded_relu_npu")

ut_case.add_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16"
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16"
    },
    ],
    "case_name": "add_case_1",
    "expect": "success"
})

ut_case.add_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32"
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32"
    },
        1.0
    ],
    "case_name": "add_case_2",
    "expect": "success"
})

ut_case.add_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NC1HWC0",
        "ori_format": "NC1HWC0",
        "dtype": "float16"
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NC1HWC0",
        "ori_format": "NC1HWC0",
        "dtype": "float16"
    },
        2
    ],
    "case_name": "add_case_3",
    "expect": "success"
})

ut_case.add_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NHWC",
        "ori_format": "NHWC",
        "dtype": "float16"
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NHWC",
        "ori_format": "NHWC",
        "dtype": "float16"
    },
        0
    ],
    "case_name": "add_case_4",
    "expect": "success"
})

ut_case.add_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NHWC",
        "ori_format": "NHWC",
        "dtype": "float16"
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NHWC",
        "ori_format": "NHWC",
        "dtype": "float16"
    },
        -1.0
    ],
    "case_name": "add_case_5",
    "expect": "failed"
})

ut_case.add_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NHWC",
        "ori_format": "NHWC",
        "dtype": "int8"
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NHWC",
        "ori_format": "NHWC",
        "dtype": "int8"
    },
    ],
    "case_name": "add_case_6",
    "expect": "failed"
})


# [TODO] coding expect function here
def np_thresholded_relu_npu(x, y, alpha=1.0):
    """input tensor(dict) : x
       output tensor(dict) : y
    """
    # input_tensor : numpy array
    input_tensor = x.get("value")
    input_tensor[input_tensor <= alpha] = 0
    return input_tensor

ut_case.add_precision_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
        "param_type": "input",
        "value_range": [-10.0, 10.0]
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
        "param_type": "output"
    },
        # alpha
        0
    ],
    "calc_expect_func": np_thresholded_relu_npu,
    "case_name": "add_precision_case_1",
})

ut_case.add_precision_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
        "param_type": "input",
        "value_range": [-10.0, 10.0]
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
        "param_type": "output"
    },
        # alpha
        2
    ],
    "calc_expect_func": np_thresholded_relu_npu,
    "case_name": "add_precision_case_2",
})

ut_case.add_precision_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
        "param_type": "input",
        "value_range": [-10.0, 10.0]
    }, {
        "shape": (1, 16, 128, 128),
        "ori_shape": (1, 16, 128, 128),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float16",
        "param_type": "output"
    },
        # alpha
    ],
    "calc_expect_func": np_thresholded_relu_npu,
    "case_name": "add_precision_case_3",
})

ut_case.add_precision_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (32,),
        "ori_shape": (32,),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32",
        "param_type": "input",
        "value_range": [-10.0, 10.0]
    }, {
        "shape": (32,),
        "ori_shape": (32,),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32",
        "param_type": "output"
    },
        # alpha
        0
    ],
    "calc_expect_func": np_thresholded_relu_npu,
    "case_name": "add_precision_case_4",
})

ut_case.add_precision_case(support_soc=["Ascend310", "Ascend610"], case={
    "params": [{
        "shape": (128,128,),
        "ori_shape": (128,128,),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32",
        "param_type": "input",
        "value_range": [-10.0, 10.0]
    }, {
        "shape": (128, 128,),
        "ori_shape": (128,128,),
        "format": "NCHW",
        "ori_format": "NCHW",
        "dtype": "float32",
        "param_type": "output"
    },
        # alpha
        5
    ],
    "calc_expect_func": np_thresholded_relu_npu,
    "case_name": "add_precision_case_5",
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})