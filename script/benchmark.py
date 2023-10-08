import json
from ppq.utils.TensorRTUtil import Benchmark, Profiling

# Benchmark(engine_file="Output/INT8.engine", steps=1000)
# Benchmark(engine_file="Output/FP32.engine", steps=1000)
rep = Profiling(engine_file="Output/INT8.engine", steps=1000)
with open("Output/INT8.profiling.json", "w") as f:
    json.dump(rep, f)
rep = Profiling(engine_file="Output/FP32.engine", steps=1000)
with open("Output/FP32.profiling.json", "w") as f:
    json.dump(rep, f)