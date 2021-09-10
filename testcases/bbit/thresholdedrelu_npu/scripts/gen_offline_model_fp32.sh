#!/bin/bash
for (( i = 31; i <= 43; i++ )); do
    ./hiai_infer_florence -i ./input_data_${i} -o ./ascend_out/ -m ./model/ThresholdedRelu_fp32_${i}.om
done
