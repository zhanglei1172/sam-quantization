import time
import requests
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import monai

from transformers import SamModel, SamProcessor

# from datasets import DownloadConfig

from gptq import *
from modelutils import *
from quant import *
from utils import *

# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open("car.png").convert("RGB")


def quantize_layer(layer, inps, nsamples, dev):
    outs = []
    if args.nearest:
        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = quantize(
                W, quantizer.scale, quantizer.zero, quantizer.maxq
            ).to(next(iter(layer.parameters())).dtype)
    for j in range(nsamples):
        outs.append(layer(inps[j : j + 1]))
    # layers[i] = layer.cpu()
    # del layer
    # outs = torch.cat(outs, dim=0)
    torch.cuda.empty_cache()
    return layer, inps, outs


@torch.no_grad()
def quantize_vision_encoder(model, testloader, dev):
    nsamples = len(testloader)

    torch.cuda.empty_cache()
    # model.patch_embed = model.patch_embed.to(dev)
    layer = model.patch_embed.to(dev)
    layer, _, inps = quantize_layer(
        layer,
        torch.cat([x["pixel_values"] for x in testloader], dim=0).to(dev),
        nsamples,
        dev,
    )
    inps = torch.cat(inps, dim=0)
    model.patch_embed = layer.cpu()
    del layer
    torch.cuda.empty_cache()

    inps += model.pos_embed

    # outs = torch.zeros_like(inps)
    layers = model.layers
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        layer, inps, outs = quantize_layer(layer, inps, nsamples, dev)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = torch.cat([out[0] for out in outs], dim=0), inps

    layer = model.neck.to(dev)
    layer, inps, vision_outputs = quantize_layer(layer, inps, nsamples, dev)
    model.neck = layer.cpu()
    del layer
    torch.cuda.empty_cache()
    return torch.cat(vision_outputs, dim=0)


@torch.no_grad()
def model_eval(model, testloader, dev):
    print("Evaluating ...")

    nsamples = len(testloader)

    # for i in range(nsamples):
    #     batch = testloader[i:i+1].to(dev)
    #     try:
    #         model(batch)
    #     except ValueError:
    #         pass
    torch.cuda.empty_cache()

    vision_outputs = quantize_vision_encoder(model.vision_encoder, testloader, dev)
    image_embeddings = vision_outputs

    dtype = next(iter(model.parameters())).dtype

    input_points = None
    input_labels = None
    input_masks = None
    input_boxes = torch.concat([x["input_boxes"] for x in testloader], dim=0).to(dev)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )
    image_positional_embeddings = model.get_image_wide_positional_embeddings()
    image_positional_embeddings = image_positional_embeddings.repeat(1, 1, 1, 1)

    low_res_masks, iou_predictions, mask_decoder_attentions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # testenc = testenc.to(dev)
    losses = []
    seg_loss = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )
    for i in range(1, nsamples):
        batch = testloader[i]
        # hidden_states = inps[i].unsqueeze(0)
        predicted_masks = low_res_masks[i]
        ground_truth_masks = batch["ground_truth_mask"].float().to(dev)
        loss = seg_loss(
            predicted_masks,
            F.interpolate(
                ground_truth_masks.unsqueeze(1).repeat(
                    1, predicted_masks.shape[1], 1, 1
                ),
                size=predicted_masks.shape[-2:],
            ),
        )
        losses.append(loss)
        masks = processor.image_processor.post_process_masks(
            predicted_masks.unsqueeze(0).cpu(),
            batch["original_sizes"].cpu(),
            batch["reshaped_input_sizes"].cpu(),
        )
        scores = iou_predictions[i : i + 1]
        show_masks_on_image(batch["original_image"][0].cpu(), masks[0], scores)
        show_boxes_on_image(
            batch["original_image"][0].cpu(),
            denormalize_box(
                processor.image_processor,
                batch["input_boxes"][0].cpu(),
                batch["original_sizes"].cpu().flatten(),
            ),
        )

        show_masks_on_image(
            batch["original_image"][0].cpu(),
            batch["ground_truth_mask"],
            scores,
            name="gt",
        )
        break
    l = torch.mean(torch.stack(losses))
    print(l.item())


@torch.no_grad()
def eval_origin(model, testloader, dev):
    losses = []
    seg_loss = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )
    nsamples = len(testloader)
    for batch in testloader[1:]:
        # forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(dev),
            input_boxes=batch["input_boxes"].to(dev),
            multimask_output=False,
        )

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(dev)
        loss = seg_loss(
            predicted_masks,
            F.interpolate(
                ground_truth_masks.unsqueeze(1).repeat(
                    1, predicted_masks.shape[1], 1, 1
                ),
                size=predicted_masks.shape[-2:],
            ),
        )
        losses.append(loss)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            batch["original_sizes"].cpu(),
            batch["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores
        # show_masks_on_image(batch["original_image"][0].cpu(), masks[0], scores)
        # show_boxes_on_image(batch["original_image"][0].cpu(), denormalize_box(processor.image_processor, batch["input_boxes"][0].cpu(), batch["original_sizes"].cpu().flatten()))

        # show_masks_on_image(batch["original_image"][0].cpu(), batch["ground_truth_mask"].repeat(1,3,1,1), scores, name="gt")
        break
    l = torch.mean(torch.stack(losses))
    print(l.item())


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="LlaMa model to load; pass location of hugginface converted checkpoint.",
    )
    parser.add_argument(
        "dataset_dir", type=str, help="Where to extract calibration data from."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--nearest", action="store_true", help="Whether to run the RTN baseline."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--sym", action="store_true", help="Whether to perform symmetric quantization."
    )
    parser.add_argument(
        "--new-eval",
        action="store_true",
        help="Whether to use the new PTB and C4 eval.",
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )

    args = parser.parse_args()
    model = SamModel.from_pretrained(
        "facebook/sam-vit-huge",
        proxies={
            "http": "socks5://127.0.0.1:7890",
            "https": "socks5://127.0.0.1:7890",
            "scoks5": "socks5://127.0.0.1:7890",
        },
    ).to(device="cuda")
    processor = SamProcessor.from_pretrained(
        "facebook/sam-vit-huge",
        proxies={
            "http": "socks5://127.0.0.1:7890",
            "https": "socks5://127.0.0.1:7890",
            "scoks5": "socks5://127.0.0.1:7890",
        },
    )
    # model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset_dir, nsamples=args.nsamples, seed=args.seed
    )
    batch = next(iter(dataloader))
    for k, v in batch.items():
        print(k, v.shape)

    # predictor = SamPredictor(sam)

    # if args.wbits < 16 and not args.nearest:
    #     tick = time.time()
    #     quantizers = llama_sequential(model, dataloader, DEV)
    #     print(time.time() - tick)

    dataloader, testloader = get_loaders(
        args.dataset_dir, nsamples=args.nsamples, seed=args.seed
    )
    # eval_origin(model, testloader, DEV)
    model_eval(model, testloader, DEV)
