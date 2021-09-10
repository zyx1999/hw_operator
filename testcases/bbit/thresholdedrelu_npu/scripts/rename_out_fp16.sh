#!/bin/bash
for (( i = 20; i <= 31; i++ )); do
    mv ./ascend_out/ThresholdedRelu_fp16_${i}_in_ThresholdedReluNpu_0:0.bin ./ascend_out/ThresholdedRelu_fp16_${i}_out.bin
done
