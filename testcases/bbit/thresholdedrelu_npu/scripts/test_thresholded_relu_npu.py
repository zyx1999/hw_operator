import os
import numpy as np
import my_utils as my

data_type = 'float16'
ALPHA_LIST_PATH = 'csv_files/designed_alpha_fp16_test.csv'
RANGE_LIST_PATH = 'csv_files/designed_range_fp16_test.csv'
GEN_SHAPE_LIST_PATH = 'csv_files/generated_shape_fp16_test.csv'

alpha_list = my.read_alpha_list_from_csv(ALPHA_LIST_PATH)
range_list = my.read_range_list_from_csv(RANGE_LIST_PATH)

shape_list = my.generate_input_shape([1, 1, 1, 1], [20, 20, 50, 100], 4, 5)

my.write_shape_list_to_csv(GEN_SHAPE_LIST_PATH, shape_list)

my.generate_onnx_models(alpha_list, range_list, shape_list, data_type)

shape_list = my.read_shape_list_from_csv(GEN_SHAPE_LIST_PATH)

my.write_input_and_expect_data_to_files(alpha_list, range_list, shape_list, data_type)