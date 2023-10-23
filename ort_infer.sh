TRT_ENGINE_PATH=Output/quantized.onnx
# TRT_ENGINE_PATH=out_q/quantized.onnx

CUDA_VISIBLE_DEVICES=3 PYTHONPATH='.' python main4.py ./checkpoints/sam_vit_h_4b8939.pth /data/seg/sbd/benchmark_RELEASE/dataset $TRT_ENGINE_PATH