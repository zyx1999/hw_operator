#!/bin/bash
model_amount=$1
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
    atc --model ../model/Thresholded_relu_npu_fp16_${order_str}.onnx --framework 5 --output ../model/Thresholded_relu_npu_fp16_${order_str} --soc_version Ascend610 --aicore_num 2
  done
fi
