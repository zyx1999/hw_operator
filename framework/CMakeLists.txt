# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/caffe_plugin")
    add_subdirectory(caffe_plugin)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx_plugin")
    add_subdirectory(onnx_plugin)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx_scope_fusion_pass")
    add_subdirectory(onnx_scope_fusion_pass)
endif()