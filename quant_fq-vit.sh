CUDA_VISIBLE_DEVICES=2 PYTHONPATH='.' python ./fq_vit/test_quant.py vit_h ./checkpoints/sam_vit_h_4b8939.pth /data/seg/sbd/benchmark_RELEASE/dataset --quant --ptf --lis --quant-method minmax --visulize