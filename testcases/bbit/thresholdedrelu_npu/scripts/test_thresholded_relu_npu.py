import os
import numpy as np
import my_utils as my

data_type = 'float16'

RANGE_LIST_PATH = 'csv_files/designed_range_fp16_1_19.csv'
GEN_ALPHA_LIST_PATH = 'csv_files/generated_alpha_fp16_1_19.csv'
GEN_SHAPE_LIST_PATH = 'csv_files/generated_shape_fp16_1_19.csv'

range_list = my.read_range_list_from_csv(RANGE_LIST_PATH)

alpha_list = my.generate_alpha_list(range_list, data_type)

my.write_alpha_list_to_csv(GEN_ALPHA_LIST_PATH, alpha_list)

shape_list = my.generate_input_shape([1,10,20,50], [20,50,100,200], 4, 19)

my.write_shape_list_to_csv(GEN_SHAPE_LIST_PATH, shape_list)

my.generate_onnx_models(alpha_list, range_list, shape_list, data_type)

my.write_input_and_expect_data_to_files(alpha_list, range_list, shape_list, data_type)