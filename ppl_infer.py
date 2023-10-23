import numpy as np

import logging
import sys
from pyppl import common as pplcommon
from pyppl import nn as pplnn

logging.basicConfig(level=logging.INFO)


class PPLModel(object):
    def __init__(self):
        self._engines = []

    def _create_x86_engine(self):
        x86_options = pplnn.x86.EngineOptions()
        x86_options.mm_policy = pplnn.x86.MM_COMPACT
        x86_engine = pplnn.x86.EngineFactory.Create(x86_options)
        if not x86_engine:
            print("create x86 engine failed.")
            sys.exit(-1)
        self._engines.append(x86_engine)

    def _create_cuda_engine(self):
        cuda_options = pplnn.cuda.EngineOptions()
        cuda_options.device_id = 0
        cuda_engine = pplnn.cuda.EngineFactory.Create(cuda_options)
        self._engines.append(cuda_engine)

    def _create_runtime(self, model_file_name):
        runtime_builder = pplnn.onnx.RuntimeBuilderFactory.Create()
        if not runtime_builder:
            logging.error("create RuntimeBuilder failed.")
            sys.exit(-1)

        status = runtime_builder.LoadModelFromFile(model_file_name)
        if status != pplcommon.RC_SUCCESS:
            logging.error(
                "init OnnxRuntimeBuilder failed: " + pplcommon.GetRetCodeStr(status)
            )
            sys.exit(-1)

        resources = pplnn.onnx.RuntimeBuilderResources()
        resources.engines = self._engines
        status = runtime_builder.SetResources(resources)
        if status != pplcommon.RC_SUCCESS:
            logging.error(
                "OnnxRuntimeBuilder SetResources failed: "
                + pplcommon.GetRetCodeStr(status)
            )
            sys.exit(-1)

        status = runtime_builder.Preprocess()
        if status != pplcommon.RC_SUCCESS:
            logging.error(
                "OnnxRuntimeBuilder preprocess failed: "
                + pplcommon.GetRetCodeStr(status)
            )
            sys.exit(-1)

        self._runtime = runtime_builder.CreateRuntime()
        if not self._runtime:
            logging.error("create Runtime instance failed.")
            sys.exit(-1)

    def _prepare_input(self, input_file):
        """set input data"""
        tensor = self._runtime.GetInputTensor(0)
        in_data = np.fromfile(input_file, dtype=np.float32).reshape((1, 3, 800, 1200))
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            logging.error(
                "copy data to tensor["
                + tensor.GetName()
                + "] failed: "
                + pplcommon.GetRetCodeStr(status)
            )
            sys.exit(-1)

    def _prepare_output(self):
        """save output"""
        for i in range(self._runtime.GetOutputCount()):
            tensor = self._runtime.GetOutputTensor(i)
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                logging.error("copy data from tensor[" + tensor.GetName() + "] failed.")
                sys.exit(-1)
            if tensor.GetName() == "dets":
                dets_data = np.array(tensor_data, copy=False)
                dets_data = dets_data.squeeze()
            if tensor.GetName() == "labels":
                labels_data = np.array(tensor_data, copy=False)
                labels_data = labels_data.squeeze()
            if tensor.GetName() == "masks":
                masks_data = np.array(tensor_data, copy=False)
                masks_data = masks_data.squeeze()
        return dets_data, labels_data, masks_data

    def run(self, engine_type, model_file_name, input_file):
        """run pplmodel

        Keyword arguments:
        engine_type -- which engine to use x86 or cuda
        model_file_name -- input model file
        input_file -- input data file (binary data)
        """
        if engine_type == "x86":
            self._create_x86_engine()
        elif engine_type == "cuda":
            self._create_cuda_engine()
        else:
            logging.error("not support engine type: ", engine_type)
            sys.exit(-1)
        self._create_runtime(model_file_name)
        self._prepare_input(input_file)
        status = self._runtime.Run()
        if status != pplcommon.RC_SUCCESS:
            logging.error("Run() failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)
        dets_data, labels_data, masks_data = self._prepare_output()
        return dets_data, labels_data, masks_data


model_file_name = "out_q/quantized.onnx"
quant_file = "out_q/quantized.json"
in_data = np.random.randn(1, 3, 1024, 1024).astype(np.float32)

ppl_model = PPLModel()

# ppl_model._create_cuda_engine()
ppl_model._create_x86_engine()
cuda_engine = ppl_model._engines[-1]
with open(quant_file, "r") as f:
    cuda_engine.Configure(pplnn.cuda.ENGINE_CONF_SET_QUANT_INFO, f.read())

ppl_model._create_runtime(model_file_name)

tensor = ppl_model._runtime.GetInputTensor(0)
status = tensor.ConvertFromHost(in_data)
ppl_model._runtime.Run()
for i in range(ppl_model._runtime.GetOutputCount()):
    tensor = ppl_model._runtime.GetOutputTensor(i)
    tensor_data = tensor.ConvertToHost()
    if not tensor_data:
        logging.error("copy data from tensor[" + tensor.GetName() + "] failed.")
        sys.exit(-1)
    dets_data = np.array(tensor_data, copy=False)
    dets_data = dets_data.squeeze()

from onnx import helper, version_converter

version_converter.convert_version(model_file_name, "out_q/converted.onnx", 11)
