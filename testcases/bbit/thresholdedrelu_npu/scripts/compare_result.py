import os
import numpy as np
import my_utils as my
# load data
GROUND_TRUTH_DIR = '../ground_truth/'
ASCEND_OUT_DIR = '../ascend_out/'
OP_NAME = 'Thresholded_relu_npu'
DATA_TYPE = '_fp16_'
GROUND_TRUTH_SUFFIX = '_gt.bin'
ASCEND_OUT_SUFFIX = '_out.bin'

for idx in range(0, 10):
    order_str = my.gen_order(idx)
    ground_truth_path = GROUND_TRUTH_DIR+OP_NAME+DATA_TYPE+order_str+GROUND_TRUTH_SUFFIX
    ascend_out_path = ASCEND_OUT_DIR+OP_NAME+DATA_TYPE+order_str+ASCEND_OUT_SUFFIX

    ground_truth_data = np.fromfile(ground_truth_path, np.float16)
    ascend_out_data = np.fromfile(ascend_out_path, np.float16)

    # compare
    my.compare_tensor(ascend_out_data, ground_truth_data, 'float16')
