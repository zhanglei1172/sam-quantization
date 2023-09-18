import time
from typing import Optional
import requests
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import monai

from transformers import SamModel, SamProcessor

# from datasets import DownloadConfig

from gptq import *
from utils.modelutils import *
from quant import *
from utils.utils import *

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
def quantize_prompt_encoder(
    model, dev, input_points=None, input_labels=None, input_boxes=None, input_masks=None
):
    layer = model.to(dev)
    torch.cuda.empty_cache()
    sparse_embeddings = None
    batch_size = 1
    target_device = layer.shared_embedding.positional_embedding.device
    if input_points is not None:
        batch_size, point_batch_size = input_points.shape[:2]
        if input_labels is None:
            raise ValueError("If points are provided, labels must also be provided.")
        point_embeddings = layer._embed_points(
            input_points, input_labels, pad=(input_boxes is None)
        )
        sparse_embeddings = point_embeddings
    if input_boxes is not None:
        batch_size = input_boxes.shape[0]
        box_embeddings = layer._embed_boxes(input_boxes)
        if sparse_embeddings is None:
            sparse_embeddings = box_embeddings
        else:
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
    if input_masks is not None:
        layer, inps, dense_embeddings = quantize_layer(
            layer.mask_embed, input_masks, len(input_masks), dev
        )
        dense_embeddings = torch.cat(dense_embeddings, dim=0)
    else:
        dense_embeddings = layer.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, layer.image_embedding_size[0], layer.image_embedding_size[1]
        )

    if sparse_embeddings is None:
        sparse_embeddings = torch.zeros(
            (batch_size, 1, 1, layer.hidden_size), device=target_device
        )
    # sparse_embeddings, dense_embeddings = layer.prompt_encoder(
    #     input_points=input_points,
    #     input_labels=input_labels,
    #     input_boxes=input_boxes,
    #     input_masks=input_masks,
    # )
    model = layer.cpu()
    del layer
    torch.cuda.empty_cache()
    return sparse_embeddings, dense_embeddings


@torch.no_grad()
def quantize_mask_decoder(
    model,
    dev,
    image_embeddings,
    image_positional_embeddings,
    sparse_prompt_embeddings,
    dense_prompt_embeddings,
    multimask_output=False,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
):
    layer = model.to(dev)
    torch.cuda.empty_cache()
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
    for j in range(len(image_embeddings)):
        outs.append(
            layer(
                image_embeddings=image_embeddings[j : j + 1],
                image_positional_embeddings=image_positional_embeddings,
                sparse_prompt_embeddings=sparse_prompt_embeddings[j : j + 1],
                dense_prompt_embeddings=dense_prompt_embeddings[j : j + 1],
                multimask_output=multimask_output,
                output_attentions=output_attentions,
                attention_similarity=attention_similarity,
                target_embedding=target_embedding,
            )
        )
    low_res_masks, iou_predictions, mask_decoder_attentions = zip(*outs)
    torch.cuda.empty_cache()

    model = layer.cpu()
    del layer
    torch.cuda.empty_cache()
    return (
        torch.cat(low_res_masks, dim=0),
        torch.cat(iou_predictions, dim=0),
        (
            None
            if output_attentions is None
            else torch.cat(mask_decoder_attentions, dim=0)
        ),
    )


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
    # sparse_embeddings, dense_embeddings = model.prompt_encoder(
    #     input_points=input_points,
    #     input_labels=input_labels,
    #     input_boxes=input_boxes,
    #     input_masks=input_masks,
    # )
    sparse_embeddings, dense_embeddings = quantize_prompt_encoder(
        model.prompt_encoder,
        dev,
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )
    image_positional_embeddings = model.get_image_wide_positional_embeddings().to(dev)
    image_positional_embeddings = image_positional_embeddings.repeat(1, 1, 1, 1)

    # low_res_masks, iou_predictions, mask_decoder_attentions = model.mask_decoder(
    #     image_embeddings=image_embeddings,
    #     image_positional_embeddings=image_positional_embeddings,
    #     sparse_prompt_embeddings=sparse_embeddings,
    #     dense_prompt_embeddings=dense_embeddings,
    #     multimask_output=False,
    # )
    low_res_masks, iou_predictions, mask_decoder_attentions = quantize_mask_decoder(
        model.mask_decoder,
        dev,
        image_embeddings,
        image_positional_embeddings,
        sparse_embeddings,
        dense_embeddings,
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
    # torch.save(model.state_dict(), "sam.pt")
    # torch.save(model.state_dict(), "./checkpoints/sam-w4.pt")
    # model.save_pretrained("./checkpoints/sam-w4")
    # save to huggingface weights:


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
    from utils.datautils import *

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
