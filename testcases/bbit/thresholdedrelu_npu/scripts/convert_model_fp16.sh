#!/bin/bash
for (( i = 20; i <= 31; i++ )); do
    atc --model ./model/Thresholded_relu_npu_fp16_${i}.onnx --framework 5 --output ./model/Thresholded_relu_npu_fp16_${i} --soc_version Ascend610 --aicore_num 2
    echo "[SUCCESS]: generate Thresholded_relu_npu_fp16_${i}.om"
done

