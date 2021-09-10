# import numpy as np
# rtol = 0.001
# atol = 0.001
# max_atol = 0.1
# alpha = 3
# actual_data = np.array([4,2,1,3])
# expect_data = np.array([4,0,0,0])
#
# a_sub_b = actual_data - expect_data
# abs_a_sub_b = np.abs(a_sub_b)
# min_abs_a_b = np.abs(expect_data)
# atol_value = min_abs_a_b*atol
# max_atol_value = min_abs_a_b * max_atol
# max_cmp = np.less_equal(abs_a_sub_b, max_atol_value)
# print("abs_a_sub_b:", abs_a_sub_b, "   max_atol_value:", max_atol_value)
# print("max_cmp:",max_cmp)
# max_cmp = max_cmp.astype(np.int8)
# print("max_cmp:",max_cmp)
# max_cmp = 1 - max_cmp
# print("max_cmp:",max_cmp)
# max_cnt = np.sum(max_cmp)
# print("max_cnt:",max_cnt)
#
# less_cmp = np.less_equal(abs_a_sub_b, atol_value)
# print("less_cmp:",less_cmp)
# less_cmp = less_cmp.astype(np.int8)
# print("less_cmp:",less_cmp)
# less_cmp = 1 - less_cmp
# print("less_cmp:",less_cmp)
# sum_cnt = np.sum(less_cmp)
# print("sum_cnt",sum_cnt)

# data_type = 'float32'
# if data_type.lower() == 'float16':
#     DATA_TYPE = 'fp16'
# elif data_type.lower() == 'float32':
#     DATA_TYPE = 'fp32'
# else:
#     print('[ERROR]: no such data type: {}'.format(data_type))
#     # return
#
# order = 13
# order_str = str(order)
# while not order // 100:
#     order_str = '0' + order_str
#     order *= 10
#
# input_data_path = "./input_data/data_{order}/Thresholded_relu_npu_{data_type}_{order}_in.bin" \
#     .format(order=order_str, data_type=DATA_TYPE)
# gt_save_path = "./ground_truth/Thresholded_relu_npu_{data_type}_{order}_in.bin" \
#     .format(order=order_str, data_type=DATA_TYPE)
#
# print(input_data_path)
# print(gt_save_path)
