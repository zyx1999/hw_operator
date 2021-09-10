#!/bin/bash
for (( i = 31; i <= 43; i++ )); do
    mv ./ascend_out/ThresholdedRelu_fp32_${i}_in_ThresholdedReluNpu_0:0.bin ./ascend_out/ThresholdedRelu_fp32_${i}_out.bin
done
