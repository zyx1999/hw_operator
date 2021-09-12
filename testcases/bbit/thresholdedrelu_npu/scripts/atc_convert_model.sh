#!/bin/bash
model_amount=$1
data_type=$2
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
    if [ ! -d "../om_model" ]; then
        mkdir "../om_model"
    fi
    if [ ${data_type}=${float16} ]; then
      atc --model ../onnx_model/Thresholded_relu_npu_fp16_${order_str}.onnx --framework 5 --output ../om_model/Thresholded_relu_npu_fp16_${order_str} --soc_version Ascend610 --aicore_num 2
    elif [ ${data_type}=${float32} ]; then
      atc --model ../onnx_model/Thresholded_relu_npu_fp32_${order_str}.onnx --framework 5 --output ../om_model/Thresholded_relu_npu_fp32_${order_str} --soc_version Ascend610 --precision_mode=allow_fp32_to_fp16 --aicore_num 2
    fi
  done
fi
