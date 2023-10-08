import numpy as np
import onnx

import tvm
from tvm import relay
from tvm.relay.frontend import from_onnx as relay_from_onnx
from tvm.relay.quantize import quantize_context, realize
from tvm.relay.transform.transform import FoldConstant, FoldScaleAxis, InferType

from tvm.relay.frontend import from_pytorch


onnx_model = onnx.load("Output/quantized.onnx")

mod, params = relay_from_onnx(
    onnx_model, opset=13, freeze_params=True, shape={"input.1": (1, 3, 1024, 1024)}
)

passes = tvm.transform.Sequential([
    relay.transform.InferType(),
    relay.transform.FakeQuantizationToInteger(),
])
mod = passes(mod)


# mod = relay.transform.InferType()(mod)
# mod = relay.transform.AnnotateTarget("int8")(mod)
# mod = relay.qnn.transform.ToQNN()(mod)

quant_passes = [InferType(), FoldConstant(), FoldScaleAxis(), realize()]
quantize_seq = tvm.transform.Sequential(quant_passes)
with tvm.transform.PassContext(
    opt_level=3,
    required_pass=["QuantizeAnnotate", "QuantizeCalibrate", "QuantizeRealize"],
):
    with quantize_context():
        mod = quantize_seq(mod)
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, params
    ).evaluate()
dtype = "float32"

tvm_output = executor(
    tvm.nd.array(np.random.randn(1, 3, 1024, 1024).astype(dtype))
).numpy()
print(1)
