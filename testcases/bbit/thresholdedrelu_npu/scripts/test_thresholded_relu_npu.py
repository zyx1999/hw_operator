import os
import numpy as np
import my_utils as my

data_type = 'float16'
alpha_list = my.read_alpha_list_from_csv('csv_files/designed_alpha_fp16_1_10.csv')
range_list = my.read_range_list_from_csv('csv_files/designed_range_fp16_1_10.csv')
shape_list = my.generate_input_shape([1, 1, 1, 1], [10, 10, 10, 10], 4, 10)
my.write_shape_list_to_csv('csv_files/generated_shape_fp16_1_10.csv', shape_list)
my.generate_onnx_models(alpha_list, range_list, shape_list, data_type)
shape_list = my.read_shape_list_from_csv('csv_files/generated_shape_fp16_1_10.csv')
my.write_input_and_expect_data_to_files(alpha_list, range_list, shape_list, data_type)