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
    if [ ! -d "./ascend_out" ]; then
        mkdir "./ascend_out"
    fi
    if [ "${data_type}" == "${float16}" ]; then
      ./hiai_infer_florence -i ./input_data/data_${order_str} -o ./ascend_out/ -m ./om_model/Thresholded_relu_npu_fp16_${order_str}.om
    fi
    if [ "${data_type}" == "${float32}" ]; then
      ./hiai_infer_florence -i ./input_data/data_${order_str} -o ./ascend_out/ -m ./om_model/Thresholded_relu_npu_fp32_${order_str}.om
    fi
  done

fi
