import os
import numpy as np
import my_utils as my
# load data
GROUND_TRUTH_DIR = './ground_truth/'
ASCEND_OUT_DIR = './ascend_out/'
OP_NAME = 'Thresholded_relu_npu'
DATA_TYPE = '_fp16_'
# DATA_TYPE = '_fp32_'
GROUND_TRUTH_SUFFIX = '_gt.bin'
ASCEND_OUT_SUFFIX = '_out.bin'
ERROR_POINTS_SUFFIX = '.csv'
dtype = 'float16'
if dtype.lower() == 'float16':
    data_type = np.float16
elif dtype.lower() == 'float32':
    data_type = np.float32
else:
    print('error!')
for idx in range(0, 12):
    order_str = my.gen_order(idx)
    ground_truth_path = GROUND_TRUTH_DIR+OP_NAME+DATA_TYPE+order_str+GROUND_TRUTH_SUFFIX
    ascend_out_path = ASCEND_OUT_DIR+OP_NAME+DATA_TYPE+order_str+ASCEND_OUT_SUFFIX
    ground_truth_data = np.fromfile(ground_truth_path, data_type)
    ascend_out_data = np.fromfile(ascend_out_path, data_type)
    error_points_file_name = OP_NAME+DATA_TYPE+order_str+ERROR_POINTS_SUFFIX
    # compare
    my.compare_tensor(ascend_out_data, ground_truth_data, dtype, error_points_file_name)
    # my.compare_tensor_bak(ascend_out_data, ground_truth_data, dtype)