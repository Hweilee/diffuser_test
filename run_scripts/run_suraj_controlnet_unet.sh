#! /bin/bash

accelerate launch suraj_train_controlnet_train_unet.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --resolution=256 \
    --learning_rate=1e-6 \
    --max_train_steps=50000 \
    --max_train_samples=3000000 \
    --dataloader_num_workers=8 \
    --validation_image=./validation  \
    --validation_prompt "a lion sitting on a park bench, high quality, 4k" "a beautiful mountain, high quality, 4k" "a dog in a park, high quality, 4k" "a man sitting on a bench, high quality, 4k" \
    --train_shards_path_or_url='pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-min512-data/{00000..01210}.tar -' \
    --proportion_empty_prompts=0.05 \
    --validation_steps=500 \
    --train_batch_size=20 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --seed=42 \
    --report_to=wandb \
    --use_8bit_adam \
    --mixed_precision=fp16 \
    --use_euler \
    --output_dir="sdxl-inpaint-controlnet-and-unet"