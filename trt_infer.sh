TRT_ENGINE_PATH=Output/INT8.engine

CUDA_VISIBLE_DEVICES=3 PYTHONPATH='.' python main4.py ./checkpoints/sam_vit_h_4b8939.pth /data/seg/sbd/benchmark_RELEASE/dataset $TRT_ENGINE_PATH