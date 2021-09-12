import numpy as np

# actual_data = np.array([4, 2, 1, 3])
# expect_data = np.array([4, 0, 0, 0])


def compare2(ground_truth_path, ascend_out_path, data_type):
    err_idx = []
    ground_truth_data = np.fromfile(ground_truth_path, data_type)
    ascend_out_data = np.fromfile(ascend_out_path, data_type)
    error_count = 0
    for index in range(ground_truth_data.shape[0]):
        gt_data = ground_truth_data[index]
        asc_out = ascend_out_data[index]
        if abs(gt_data - asc_out) > min(abs(gt_data), abs(asc_out)) * 0.001:
            error_count += 1
            err_idx.append(index)
    error_rate = error_count / ground_truth_data.shape[0]
    if error_rate > 0.001:
        print("Compare Failed: error_count/all = {}/{}".format(
            error_count, ground_truth_data.shape[0]))
    else:
        print("Compare Success: error_count/all = {}/{}".format(
            error_count, ground_truth_data.shape[0]))
    print()

def compare1(actual_data, expect_data):
    rtol = 0.001
    atol = 0.001
    max_atol = 0.1

    def _compare_tensor():
        a_sub_b = actual_data - expect_data
        abs_a_sub_b = np.abs(a_sub_b)
        min_abs_a_b = np.abs(expect_data)
        atol_value = min_abs_a_b * atol
        max_cnt = 0
        if max_atol:
            max_atol_value = min_abs_a_b * max_atol
            max_cmp = np.less_equal(abs_a_sub_b, max_atol_value)
            # print("abs_a_sub_b:", abs_a_sub_b, "   max_atol_value:", max_atol_value)
            # print("max_cmp:", max_cmp)
            max_cmp = max_cmp.astype(np.int8)
            # print("max_cmp:", max_cmp)
            max_cmp = 1 - max_cmp
            # print("max_cmp:", max_cmp)
            max_cnt = np.sum(max_cmp)
            # print("max_cnt:", max_cnt)

        less_cmp = np.less_equal(abs_a_sub_b, atol_value)
        # print("less_cmp:", less_cmp)
        less_cmp = less_cmp.astype(np.int8)
        # print("less_cmp:", less_cmp)
        less_cmp = 1 - less_cmp
        # print("less_cmp:", less_cmp)
        sum_cnt = np.sum(less_cmp)
        # print("sum_cnt", sum_cnt)
        return sum_cnt, max_cnt

    rtol_cnt, max_atol_cnt = _compare_tensor()
    # print(rtol_cnt, max_atol_cnt, rtol * expect_data.size)
    is_success = True
    err_msg = ''
    if rtol_cnt > rtol * expect_data.size:
        is_success = False
        err_msg = "Error count (expect - actual > atol * expect): %s, rtol is %s, total size: %s." \
                  % (str(rtol_cnt), str(rtol), str(expect_data.size))
    if max_atol_cnt > 0:
        is_success = False
        err_msg += "Max atol error count(expect - acutal > max_atol * expect): %d, max_atol is: %s" \
                   % (max_atol_cnt, str(max_atol))
    if is_success:
        print('[SUCCESS]: compare success.')
    else:
        print('[FAILED]: ' + err_msg)
