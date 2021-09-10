#!/bin/bash
for (( i = 31; i <= 43; i++ )); do
    atc --model ./model/ThresholdedRelu_fp32_${i}.onnx --framework 5 --output ./model/ThresholdedRelu_fp32_${i} --soc_version Ascend610 --precision_mode=allow_fp32_to_fp16 --aicore_num 2
    echo "[SUCCESS]: generate ThresholdedRelu_fp32_${i}.om"
done
