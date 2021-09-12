#!/bin/bash
model_amount=$1
float16='fp16'
float32='fp32'
if [ ${model_amount} -gt 0 ]; then
  for (( i = 0; i < $1; i++ ));
  do
    order=$[ ${i} + 1 ]
    order_str=""${order}
    zero="0"
    while [ $[${order}/100] == 0 ];
    do
        order_str=${zero}${order_str}
        order=$[ $order * 10 ]
    done
    echo ${order_str}
    if [ ${data_type}=${float16} ]; then
      mv ../ascend_out/Thresholded_relu_npu_fp16_${order_str}_in_ThresholdedReluNpu_0:0.bin ../ascend_out/Thresholded_relu_npu_fp16_${order_str}_out.bin
    elif [ ${data_type}=${float32} ]; then
      mv ../ascend_out/Thresholded_relu_npu_fp32_${order_str}_in_ThresholdedReluNpu_0:0.bin ../ascend_out/Thresholded_relu_npu_fp32_${order_str}_out.bin
    fi
  done
fi
