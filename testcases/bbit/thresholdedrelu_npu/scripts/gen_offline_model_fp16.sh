#!/bin/bash
for (( i = 1; i <= 10; i++ ));
do
    ./hiai_infer_florence -i ./input_datas/input_data_${i} -o ./ascend_out/ -m ./model/ThresholdedRelu_fp16_${i}.om
done
