import json
from ppq.utils.TensorRTUtil import Benchmark, Profiling

steps = 1000

# Benchmark(engine_file="Output/INT8.engine", steps=1000)
# Benchmark(engine_file="Output/FP32.engine", steps=1000)


b_int8 = Benchmark(engine_file="Output/INT8.engine", steps=steps) / steps
b_fp16 = Benchmark(engine_file="Output/FP16.engine", steps=steps) / steps
b_fp32 = Benchmark(engine_file="Output/FP32.engine", steps=steps) / steps
print(f"INT8: {b_int8}s, FP16: {b_fp16}s, FP32: {b_fp32}s")

# rep = Profiling(engine_file="Output/INT8.engine", steps=1000)
# with open("Output/INT8.profiling.json", "w") as f:
#     json.dump(rep, f)
# rep = Profiling(engine_file="Output/FP32.engine", steps=1000)
# with open("Output/FP32.profiling.json", "w") as f:
#     json.dump(rep, f)
