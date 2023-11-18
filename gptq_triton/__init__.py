import itertools
import json
from pathlib import Path
from typing import Optional

import torch

from . import fused_mlp, quant_linear
from .fused_attention import QuantAttention, make_quant_attn

# from .fused_mlp import make_fused_mlp
from .quant_linear import QuantLinear, make_quant, triton_matmul4


def load_quant(
    model,
    checkpoint: str,
    warmup_autotune: bool = True,
    device: Optional[str] = "cuda",
    fuse_mlp: Optional[bool] = None,
    sub_module: Optional[str] = None,
):
    """
    Load a quantized model from a checkpoint.
    Args:
            checkpoint: Path to the checkpoint directory.
            warmup_autotune: If True, run a warmup autotune pass. Otherwise autotune will run during forward passes.
            device: Device to run the model on; needed if warmup_autotune is True.
            fuse_mlp: If True, replace the MLP layers with fused versions.  If None, will apply fuse_mlp if the model's groupsize is -1, otherwise fuse_mlp will be disabled (it's slower when using grouping).
    Returns:
            The loaded model.
    """
    quant_config = json.load(open(Path(checkpoint) / "quant_config.json"))
    wbits = quant_config["wbits"]
    groupsize = quant_config["groupsize"]
    if sub_module:
        model_quant = model.__getattr__(sub_module)
    else:
        model_quant = model

    make_quant(model_quant, wbits, groupsize)

    # Load the quantized checkpoint
    print("Loading model ...")
    if (Path(checkpoint) / "model.safetensors").exists():
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(Path(checkpoint) / "model.safetensors"))
    elif (Path(checkpoint) / "model.pt").exists():
        model.load_state_dict(torch.load(Path(checkpoint) / "model.pt"), strict=False)
    else:
        raise FileNotFoundError(
            f"Could not find model checkpoint at {checkpoint}; please ensure that the path is correct and contains a `model.pt` or `model.safetensors` file."
        )

    # Go through all the QuantLinear layers and if their bias is all zeros, set it to None
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            if m.bias is not None and (m.bias == 0).all():
                m.bias = None
                # print(f"Removed bias from {name}")

    make_quant_attn(model_quant)

    if fuse_mlp == True or (fuse_mlp is None and groupsize == -1):
        make_fused_mlp(model_quant)

    # Move the model to the correct device
    if device is not None:
        model = model.to(device)

    # Warm up the autotune cache
    if warmup_autotune:
        if device is None:
            raise ValueError("You must specify a device when warmup_autotune is True.")

        autotune_warmup(model)

    print("Done.")

    return model


def autotune_warmup(model):
    """
    The Triton kernels autotune themselves for specific input sizes.  But this takes time.
    This function collects information on all possible input sizes for the different kernels
    and then runs them through the autotuner.
    The intended use is to run this on startup so the autotuner doesn't have to run during
    actual inference.
    """
    from tqdm import tqdm

    warmups = itertools.chain(
        quant_linear.autotune_warmup(model)
    )
    warmups = list(warmups)

    print("Warming up autotune cache ...")
    with torch.no_grad():
        for m in tqdm(range(0, 12)):
            m = 2**m
            for func in warmups:
                func(m)
