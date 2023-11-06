# This implementation is based on this one
# https://github.com/Pie31415/diffusers/blob/vae-training/examples/vae/train_vae.py
import argparse
import math
import os
import shutil
from datetime import datetime
from typing import List

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image as PIL_Image
from datasets import load_dataset,load_from_disk
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_bitsandbytes_available
from src.models import AutoencoderSIS


if is_wandb_available():
    import wandb
    os.environ['WANDB_START_METHOD']="thread"

def parse_bool(str: str):
    return str.upper() == "TRUE"


logger = get_logger(__name__, log_level="INFO")
parser = argparse.ArgumentParser(description="VAE training script.")
# Data Management
parser.add_argument("--dataset_name_or_path", type=str, default="/mnt/c/BUSDATA/Datasets/ShearoIA/hf_all/")
parser.add_argument("--dataset_img_height", type=int, default=64)
parser.add_argument("--dataset_img_width", type=int, default=128)
parser.add_argument("--dataset_img_depth", type=int, default=2)
parser.add_argument("--vae_spacial_compression", type=int, default=4, help="Should be a power of 2")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/output/vae",
    help="The output directory where the model predictions and checkpoints will be written.",
)
# Training Management
parser.add_argument("--train_tiling_ratio", type=int, default=1)
parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--max_train_steps", type=int, default=10)
parser.add_argument("--vae_train_sampling", type=parse_bool, default="True")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--scale_lr",
    action="store_true",
    default=False,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="constant",
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
)
parser.add_argument(
    "--lr_warmup_steps",
    type=int,
    default=500,
    help="Number of steps for the warmup in the lr scheduler.",
)
parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
parser.add_argument("--optim_weight_decay", type=float, default=1e-2, help="optimizer weight decay like in paper")
parser.add_argument("--optim_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="fp16",
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)
parser.add_argument(
    "--lambda_kl",
    type=float,
    default=1e-6,
    help="Scaling factor for the Kullback-Leibler divergence penalty term.",
)
parser.add_argument(
    "--lambda_lpips",
    type=float,
    default=1e-1,
    help="Scaling factor for the LPIPS metric",
)
# Validation Management
parser.add_argument("--val_every_nepochs", type=int, default=1, help="Number of training epochs before validation...")
parser.add_argument("--val_num_samples", type=int, default=10)
parser.add_argument(
    "--tracker_name",
    type=str,
    default=["mlflow", "wandb"],
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    ),
)
parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Indicate the checkpoint frequency")
parser.add_argument(
    "--checkpointing_total_limit", type=int, default=10, help="Indicate the max number of checkpoints to keep"
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default="latest",
    help="Indicate either the checkpoint directory or the latest",
)
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
parser.add_argument("--debug", action="store_true", default=False)


def main(
    dataset_name_or_path: str,
    dataset_img_height: int,
    dataset_img_width:int,
    dataset_img_depth:int,
    vae_spacial_compression: int,
    output_dir: str,
    train_tiling_ratio:int,
    train_batch_size: int,
    max_train_steps: int,
    vae_train_sampling: bool,
    gradient_accumulation_steps: int,
    gradient_checkpointing: int,
    learning_rate: float,
    scale_lr: bool,
    lr_scheduler: str,
    lr_warmup_steps: int,
    use_8bit_adam: bool,
    optim_weight_decay: float,
    optim_max_grad_norm: float,
    mixed_precision: str,
    lambda_kl: float,
    lambda_lpips: float,
    val_every_nepochs: int,
    val_num_samples: int,
    tracker_name: List[str],
    checkpointing_steps: int,
    checkpointing_total_limit: int,
    resume_from_checkpoint: str,
    seed: int,
    debug: bool = False,
    all_args: dict = None,
):
    if debug:
        NMAX = 100
    else:
        NMAX = None
    # We create the logging directory...
    logdir_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-SISModel"
    logdir_path = os.path.join(output_dir, logdir_name)
    project_dir = output_dir
    if resume_from_checkpoint.upper() == "NONE":
        # We won't from checkpoint then we'll create the logdir to save the model...
        project_dir = logdir_path
    print(f"Project Directory : {project_dir}")
    # Configure Accelerate
    accelerator_project_configuration = ProjectConfiguration(project_dir=project_dir, logging_dir=logdir_path)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=tracker_name,
        project_config=accelerator_project_configuration,
    )
    if accelerator.scaler is not None:
        accelerator.scaler.set_growth_interval(100)
    if accelerator.is_main_process:
        os.makedirs(logdir_path, exist_ok=True)

    # Manage Seed
    set_seed(seed)
    # We create a Dataset and Dataloaders
    if os.path.exists(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
    # We create train / val dataset...
    train_dataset = dataset['train']
    if NMAX:
        train_indices = list(range(min(len(train_dataset),NMAX)))
    else:
        train_indices = list(range(len(train_dataset)))
    test_dataset = dataset['test']
    test_indices = list(range(min(len(train_dataset),val_num_samples)))
    # We crop with respect to tiling ratio... 
    crop_ratio = 1.0/train_tiling_ratio
    train_height = dataset_img_height//train_tiling_ratio
    train_width = dataset_img_width//train_tiling_ratio
    def train_transform(examples):
        transforms_img = A.Compose([
            A.RandomResizedCrop(
                height=train_height,
                width=train_width,
                scale=(crop_ratio/2,crop_ratio)),
            A.Normalize(0.5,0.5),
            ToTensorV2()
        ])
        # First, we convert to multilayered images
        if 'video_img' in examples:
            examples["image"]=[np.stack((np.array(s),np.array(v)),axis=-1) for v,s in zip(examples["video_img"],examples['shearo_img'])]
            del examples["video_img"],examples["shearo_img"]
        else:
            examples['image']=[np.array(image.convert('RGB')) for image in examples['image']]
        examples["augmented"] = [transforms_img(image=image,mask=np.array(mask)) for image,mask in zip(examples['image'],examples['annotation'])]
        examples["annotation"] = [aug['mask'] for aug in examples["augmented"]]
        examples["image"] = [aug['image'] for aug in examples["augmented"]]
        del examples["augmented"]
        return examples

    def test_transform(examples):
        transforms_img = A.Compose([
            A.Resize(dataset_img_height,dataset_img_width),
            A.Normalize(0.5,0.5),
            ToTensorV2()
        ])
        # First, we convert to multilayered images
        if 'video_img' in examples:
            examples["image"]=[np.stack((np.array(s),np.array(v)),axis=-1) for v,s in zip(examples["video_img"],examples['shearo_img'])]
            del examples["video_img"],examples["shearo_img"]
        else:
            examples['image']=[np.array(image.convert('RGB')) for image in examples['image']]
        examples["augmented"] = [transforms_img(image=image,mask=np.array(mask)) for image,mask in zip(examples['image'],examples['annotation'])]
        examples["annotation"] = [aug['mask'] for aug in examples["augmented"]]
        examples["image"] = [aug['image'] for aug in examples["augmented"]]
        del examples["augmented"]
        return examples

    train_dataset.set_transform(train_transform)
    test_dataset.set_transform(test_transform)

    train_dataloader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), train_batch_size),
    )
    val_dataloader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    config = AutoencoderSIS.get_config(
        sample_size=max(dataset_img_height,dataset_img_width),
        compression=vae_spacial_compression,
        in_channels=dataset_img_depth,
        out_channels=dataset_img_depth,
    )
    vae = AutoencoderSIS(**config)
    vae: AutoencoderSIS
    vae.requires_grad_(True)

    # In order to save model correctly...
    def save_model_hook(models: List[AutoencoderSIS], weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "vae"))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models: List[AutoencoderSIS], input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            # load diffusers style into model
            load_model = AutoencoderSIS.from_pretrained(input_dir, subfolder="vae")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
    if is_bitsandbytes_available() and use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(vae.parameters(), lr=learning_rate, weight_decay=optim_weight_decay)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Prepare everything with our `accelerator`.
    (
        vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(vae, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("train-vae-CelebA", all_args, init_kwargs={"wandb": {"dir": "./outputs"}})

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_indices)}")
    logger.info(f"  Num test samples = {len(test_indices)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)
    total_loss = 0.0
    kl_loss = 0.0
    lpips_loss = 0.0
    mse_loss = 0.0
    for epoch in range(first_epoch, num_train_epochs):
        vae.train()
        accelerator.wait_for_everyone()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                total_loss_batch = 0.0
                x = batch['image']
                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                # We use a "trick" in order to use DDP that needs a forward method.
                posterior = vae.forward(x, encode=True).latent_dist
                # We sample during training instead of mode..
                if vae_train_sampling:
                    z = posterior.sample()
                else:
                    z = posterior.mode()
                pred = vae.forward(z, decode=True).sample
                kl_loss_batch = posterior.kl().mean()
                mse_loss_batch = F.mse_loss(pred, x, reduction="mean")
                if x.shape[1]<3:
                    # We'll put zeros in the third channel...
                    n_to_add = 3-x.shape[1]
                    x_pad = torch.nn.functional.pad(x,(0,0,0,0,0,n_to_add),mode='constant',value=0)
                    pred_pad = torch.nn.functional.pad(pred,(0,0,0,0,0,n_to_add),mode='constant',value=0)
                    lpips_loss_batch = lpips_loss_fn(pred_pad, x_pad).mean()
                elif x.shape[1]>3:
                    lpips_loss_batch = lpips_loss_fn(pred[:,:3,:,:], x[:,:3,:,:]).mean()
                else:
                    lpips_loss_batch = lpips_loss_fn(pred, x).mean()


                total_loss_batch = mse_loss_batch + lambda_lpips * lpips_loss_batch + lambda_kl * kl_loss_batch

                # Gather the losses across all processes for logging (if we use distributed training).
                total_loss += (
                    accelerator.gather(total_loss_batch.repeat(train_batch_size)).mean() / gradient_accumulation_steps
                )
                kl_loss += (
                    accelerator.gather(kl_loss_batch.repeat(train_batch_size)).mean() / gradient_accumulation_steps
                )
                lpips_loss += (
                    accelerator.gather(lpips_loss_batch.repeat(train_batch_size)).mean() / gradient_accumulation_steps
                )
                mse_loss += (
                    accelerator.gather(mse_loss_batch.repeat(train_batch_size)).mean() / gradient_accumulation_steps
                )

                accelerator.backward(total_loss_batch)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), optim_max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "step_loss": total_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "mse": mse_loss.detach().item(),
                    "lpips": lpips_loss.detach().item(),
                    "kl": kl_loss.detach().item(),
                }
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
                total_loss = 0.0
                mse_loss = 0.0
                kl_loss = 0.0
                lpips_loss = 0.0
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if checkpointing_total_limit is not None:
                            checkpoints = os.listdir(project_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= checkpointing_total_limit:
                                num_to_remove = len(checkpoints) - checkpointing_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(project_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(project_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= max_train_steps:
                print(f'global_step ({global_step}) > max_train_steps ({max_train_steps})')
                break

        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            if epoch % val_every_nepochs == 0:
                with torch.no_grad():
                    logger.info("Running validation... ")
                    vae_model = accelerator.unwrap_model(vae)
                    vae_model.eval()
                    images = {'images':[],'images_shearo':[],'images_video':[]}
                    image_ids = []
                    for _, batch in enumerate(val_dataloader):
                        ids = batch['image_id']
                        x = batch['image']
                        reconstructions = vae_model(x).sample
                        th_arr = torch.cat([x.cpu(), reconstructions.cpu()], axis=-1).squeeze(0)  # Last dim = Width
                        np_img = (127.5 * (th_arr + 1)).permute(1, 2, 0).numpy().astype(np.uint8)
                        if dataset_img_depth<3:
                            images['images_video'].append(PIL_Image.fromarray(np_img[:,:,0], mode="L")) # Layer 0 => Video
                            images['images_shearo'].append(PIL_Image.fromarray(np_img[:,:,1], mode="L")) # Layer 1 => Shearo
                        else:
                            images['images'].append(PIL_Image.fromarray(np_img.convert("RGB"))) 
                        image_ids.append(ids)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            for key,values in images.items():
                                if len(values)>0:
                                    th_images = torch.stack([torch.tensor(np.array(img)) for img in values])
                                    while th_images.ndim<4:
                                        th_images = th_images.unsqueeze(-1)
                                    tracker.writer.add_images(f"Original (left) / Reconstruction (right) {key}", th_images, epoch, dataformats="NHWC")
                        elif tracker.name == "wandb":
                            wandb_images = []
                            for key,values in images.items():
                                if len(values)>0:
                                    wandb_images.extend([
                                            wandb.Image(image, caption=f"{i}: {key}{image_ids[i]}")
                                            for i, image in enumerate(values)
                                        ])
                            tracker.log({"validation": wandb_images})
                    del vae_model
                    torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(os.path.join(output_dir, "vae_final"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["all_args"] = args_dict.copy()
    main(**args_dict)
