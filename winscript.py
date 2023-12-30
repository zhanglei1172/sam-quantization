import glob
import subprocess
import os



srcs = glob.glob('*.cu')

for src in srcs:
    command = '''-D_DISABLE_EXTENDED_ALIGNED_STORAGE -t=0 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -w -gencode="arch=compute_80,code=[sm_80,compute_80]" -O3 -std=c++17 --expt-relaxed-constexpr -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG --use_fast_math -DAIT_USE_FAST_MATH=1 -Xcompiler=-fPIC -Xcompiler=-fno-strict-aliasing -Xcompiler -fvisibility=hidden -I../../3rdparty/../static/include/kernels -I../../3rdparty/cutlass/include -I../../3rdparty/cutlass/tools/util/include -I../../3rdparty/cutlass/examples/35_gemm_softmax -I../../3rdparty/cutlass/examples/41_fused_multi_head_attention -I../../3rdparty/cutlass/examples/45_dual_gemm -I../../3rdparty/cutlass/../../python/aitemplate/backend/cuda/attention/src/./ -I../../3rdparty/cutlass/../../python/aitemplate/backend/cuda/attention/src/fmha  -Id:/Users/islei/miniconda3/envs/com/include -c -o'''
    # ret_code = subprocess.call(['nvcc'] + command.split() + [src.replace('.cu', '.obj'), src])
    ret_code = os.system(" ".join(['nvcc'] + command.split() + [src.replace('.cu', '.obj'), src]))
    if ret_code != 0:
        print('Failed to compile %s' % src)
        exit(ret_code)
        