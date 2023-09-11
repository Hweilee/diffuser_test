import argparse
import os

import torch

from diffusers import (
    KarrasEDMPipeline,
    KarrasEDMScheduler,
    UNet2DModel,
)


EDM_TEST_UNET_CONFIG = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "time_embedding_type": "positional",
    "freq_shift": 0,
    "flip_sin_to_cos": True,
    "layers_per_block": 2,
    "num_class_embeds": 1000,
    "block_out_channels": [64, 128],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",  # TODO: want downsample to be resnet w/ conv downsample at end??
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
    "act_fn": "silu",
}

CM_TEST_UNET_CONFIG = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "num_class_embeds": 1000,
    "block_out_channels": [32, 64],
    "attention_head_dim": 8,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

EDM_IMAGENET_64_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 3,
    "num_class_embeds": 1000,
    "block_out_channels": [192, 192 * 2, 192 * 3, 192 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

CM_IMAGENET_64_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 3,
    "num_class_embeds": 1000,
    "block_out_channels": [192, 192 * 2, 192 * 3, 192 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

CM_LSUN_256_UNET_CONFIG = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "num_class_embeds": None,
    "block_out_channels": [256, 256, 256 * 2, 256 * 2, 256 * 4, 256 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "default",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

EDM_SCHEDULER_CONFIG = {
    "num_train_timesteps": 256,
    "sigma_min": 0.05,
    "sigma_max": 50.0,
    "s_churn": 40.0,
    "s_noise": 1.003,
}

CM_SCHEDULER_CONFIG = {
    "num_train_timesteps": 40,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "s_churn": 0.0,
}


def convert_cm_edm_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=False):
    new_checkpoint[f"{new_prefix}.norm1.weight"] = checkpoint[f"{old_prefix}.in_layers.0.weight"]
    new_checkpoint[f"{new_prefix}.norm1.bias"] = checkpoint[f"{old_prefix}.in_layers.0.bias"]
    new_checkpoint[f"{new_prefix}.conv1.weight"] = checkpoint[f"{old_prefix}.in_layers.2.weight"]
    new_checkpoint[f"{new_prefix}.conv1.bias"] = checkpoint[f"{old_prefix}.in_layers.2.bias"]
    new_checkpoint[f"{new_prefix}.time_emb_proj.weight"] = checkpoint[f"{old_prefix}.emb_layers.1.weight"]
    new_checkpoint[f"{new_prefix}.time_emb_proj.bias"] = checkpoint[f"{old_prefix}.emb_layers.1.bias"]
    new_checkpoint[f"{new_prefix}.norm2.weight"] = checkpoint[f"{old_prefix}.out_layers.0.weight"]
    new_checkpoint[f"{new_prefix}.norm2.bias"] = checkpoint[f"{old_prefix}.out_layers.0.bias"]
    new_checkpoint[f"{new_prefix}.conv2.weight"] = checkpoint[f"{old_prefix}.out_layers.3.weight"]
    new_checkpoint[f"{new_prefix}.conv2.bias"] = checkpoint[f"{old_prefix}.out_layers.3.bias"]

    if has_skip:
        new_checkpoint[f"{new_prefix}.conv_shortcut.weight"] = checkpoint[f"{old_prefix}.skip_connection.weight"]
        new_checkpoint[f"{new_prefix}.conv_shortcut.bias"] = checkpoint[f"{old_prefix}.skip_connection.bias"]

    return new_checkpoint


def convert_cm_edm_attention(checkpoint, new_checkpoint, old_prefix, new_prefix, attention_dim=None):
    weight_q, weight_k, weight_v = checkpoint[f"{old_prefix}.qkv.weight"].chunk(3, dim=0)
    bias_q, bias_k, bias_v = checkpoint[f"{old_prefix}.qkv.bias"].chunk(3, dim=0)

    new_checkpoint[f"{new_prefix}.group_norm.weight"] = checkpoint[f"{old_prefix}.norm.weight"]
    new_checkpoint[f"{new_prefix}.group_norm.bias"] = checkpoint[f"{old_prefix}.norm.bias"]

    new_checkpoint[f"{new_prefix}.to_q.weight"] = weight_q.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_q.bias"] = bias_q.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_k.weight"] = weight_k.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_k.bias"] = bias_k.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_v.weight"] = weight_v.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_v.bias"] = bias_v.squeeze(-1).squeeze(-1)

    new_checkpoint[f"{new_prefix}.to_out.0.weight"] = (
        checkpoint[f"{old_prefix}.proj_out.weight"].squeeze(-1).squeeze(-1)
    )
    new_checkpoint[f"{new_prefix}.to_out.0.bias"] = checkpoint[f"{old_prefix}.proj_out.bias"].squeeze(-1).squeeze(-1)

    return new_checkpoint


def convert_cm_edm_to_diffusers(checkpoint_path: str, unet_config):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = checkpoint["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = checkpoint["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = checkpoint["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = checkpoint["time_embed.2.bias"]

    if unet_config["num_class_embeds"] is not None:
        new_checkpoint["class_embedding.weight"] = checkpoint["label_emb.weight"]

    new_checkpoint["conv_in.weight"] = checkpoint["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = checkpoint["input_blocks.0.0.bias"]

    down_block_types = unet_config["down_block_types"]
    layers_per_block = unet_config["layers_per_block"]
    attention_head_dim = unet_config["attention_head_dim"]
    channels_list = unet_config["block_out_channels"]
    current_layer = 1
    prev_channels = channels_list[0]

    for i, layer_type in enumerate(down_block_types):
        current_channels = channels_list[i]
        downsample_block_has_skip = current_channels != prev_channels
        if layer_type == "ResnetDownsampleBlock2D":
            for j in range(layers_per_block):
                new_prefix = f"down_blocks.{i}.resnets.{j}"
                old_prefix = f"input_blocks.{current_layer}.0"
                has_skip = True if j == 0 and downsample_block_has_skip else False
                new_checkpoint = convert_cm_edm_resnet(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=has_skip
                )
                current_layer += 1

        elif layer_type == "AttnDownBlock2D":
            for j in range(layers_per_block):
                new_prefix = f"down_blocks.{i}.resnets.{j}"
                old_prefix = f"input_blocks.{current_layer}.0"
                has_skip = True if j == 0 and downsample_block_has_skip else False
                new_checkpoint = convert_cm_edm_resnet(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=has_skip
                )
                new_prefix = f"down_blocks.{i}.attentions.{j}"
                old_prefix = f"input_blocks.{current_layer}.1"
                new_checkpoint = convert_cm_edm_attention(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, attention_head_dim
                )
                current_layer += 1

        if i != len(down_block_types) - 1:
            new_prefix = f"down_blocks.{i}.downsamplers.0"
            old_prefix = f"input_blocks.{current_layer}.0"
            new_checkpoint = convert_cm_edm_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)
            current_layer += 1

        prev_channels = current_channels

    # hardcoded the mid-block for now
    new_prefix = "mid_block.resnets.0"
    old_prefix = "middle_block.0"
    new_checkpoint = convert_cm_edm_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)
    new_prefix = "mid_block.attentions.0"
    old_prefix = "middle_block.1"
    new_checkpoint = convert_cm_edm_attention(checkpoint, new_checkpoint, old_prefix, new_prefix, attention_head_dim)
    new_prefix = "mid_block.resnets.1"
    old_prefix = "middle_block.2"
    new_checkpoint = convert_cm_edm_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)

    current_layer = 0
    up_block_types = unet_config["up_block_types"]

    for i, layer_type in enumerate(up_block_types):
        if layer_type == "ResnetUpsampleBlock2D":
            for j in range(layers_per_block + 1):
                new_prefix = f"up_blocks.{i}.resnets.{j}"
                old_prefix = f"output_blocks.{current_layer}.0"
                new_checkpoint = convert_cm_edm_resnet(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=True
                )
                current_layer += 1

            if i != len(up_block_types) - 1:
                new_prefix = f"up_blocks.{i}.upsamplers.0"
                old_prefix = f"output_blocks.{current_layer-1}.1"
                new_checkpoint = convert_cm_edm_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)
        elif layer_type == "AttnUpBlock2D":
            for j in range(layers_per_block + 1):
                new_prefix = f"up_blocks.{i}.resnets.{j}"
                old_prefix = f"output_blocks.{current_layer}.0"
                new_checkpoint = convert_cm_edm_resnet(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, has_skip=True
                )
                new_prefix = f"up_blocks.{i}.attentions.{j}"
                old_prefix = f"output_blocks.{current_layer}.1"
                new_checkpoint = convert_cm_edm_attention(
                    checkpoint, new_checkpoint, old_prefix, new_prefix, attention_head_dim
                )
                current_layer += 1

            if i != len(up_block_types) - 1:
                new_prefix = f"up_blocks.{i}.upsamplers.0"
                old_prefix = f"output_blocks.{current_layer-1}.2"
                new_checkpoint = convert_cm_edm_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix)

    new_checkpoint["conv_norm_out.weight"] = checkpoint["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = checkpoint["out.0.bias"]
    new_checkpoint["conv_out.weight"] = checkpoint["out.2.weight"]
    new_checkpoint["conv_out.bias"] = checkpoint["out.2.bias"]

    return new_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_path", type=str, default=None, required=True, help="Path to the unet.pt to convert.")
    parser.add_argument(
        "--dump_path", type=str, default=None, required=True, help="Path to output the converted UNet model."
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="edm",
        help="The implementation of the checkpoint. Currently 'edm' and 'cm_edm' are supported.",
    )
    parser.add_argument("--class_cond", action="store_true", help="Whether the model is class-conditional.")

    args = parser.parse_args()

    ckpt_name = os.path.basename(args.unet_path)
    print(f"Checkpoint: {ckpt_name}")

    if args.implementation not in ["edm", "cm_edm"]:
        raise ValueError(
            f"Implementation type {args.implementation} is not supported. Currently supported implementation types are"
            " 'edm' and 'cm_edm'."
        )

    # Get U-Net config
    if args.implementation == "edm":
        if "test" in ckpt_name:
            unet_config = EDM_TEST_UNET_CONFIG
        else:
            unet_config = EDM_IMAGENET_64_UNET_CONFIG
    elif args.implementation == "cm_edm":
        if "imagenet64" in ckpt_name:
            unet_config = CM_IMAGENET_64_UNET_CONFIG
        elif "256" in ckpt_name and (("bedroom" in ckpt_name) or ("cat" in ckpt_name)):
            unet_config = CM_LSUN_256_UNET_CONFIG
        elif "test" in ckpt_name:
            unet_config = CM_TEST_UNET_CONFIG
        else:
            raise ValueError(f"CM checkpoint type {ckpt_name} is not currently supported.")

    if not args.class_cond:
        unet_config["num_class_embeds"] = None

    if args.implementation == "edm":
        raise NotImplementedError("Converting EDM checkpoints is not yet supported.")
    elif args.implementation == "cm_edm":
        converted_unet_ckpt = convert_cm_edm_to_diffusers(args.unet_path, unet_config)

    image_unet = UNet2DModel(**unet_config)
    image_unet.load_state_dict(converted_unet_ckpt)

    # Get scheduler config
    if args.implementation == "edm":
        scheduler_config = EDM_SCHEDULER_CONFIG
    elif args.implementation == "cm_edm":
        scheduler_config = CM_SCHEDULER_CONFIG

    edm_scheduler = KarrasEDMScheduler(**scheduler_config)

    consistency_model = KarrasEDMPipeline(unet=image_unet, scheduler=edm_scheduler)
    consistency_model.save_pretrained(args.dump_path)
