import numpy as np

def compare(gt_path, ascend_out_path, dtype = np.float32):

    gt_data = np.fromfile(gt_path, dtype)
    ascend_out_data = np.fromfile(ascend_out_path, dtype)

    diff = np.abs(ascend_out_data - gt_data)
    max_diff_idx = np.argmax(diff)
    eps = 1e-2
    
    error_count = np.sum(diff>eps)
    error_rate = error_count / (gt_data.shape[0])
    
    if error_count > 0:
        print("[Compare Failed]: error rate: {:.2f}% = {}/{}".format(error_rate*100,error_count,gt_data.shape[0]))
    else:
        print("Compare Success!")

    print("Max diff: ground truth:{}, ascend output:{}".format(gt_data[max_diff_idx],ascend_out_data[max_diff_idx]))


if __name__ == "__main__":
    gt_path = "./ground_truth/abs_npu_gt_1_16_128_128.bin"
    ascend_out_path = "./ascend_out/out.bin"
    compare(gt_path, ascend_out_path)
