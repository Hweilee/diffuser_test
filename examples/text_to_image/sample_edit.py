from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
from utils import save_image, concat_images_in_square_grid, TaskVector
import argparse

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', type=str, default="stabilityai/stable-diffusion-2", help='pretrained model')
    parser.add_argument('--model_finetuned', type=str, default="", help='finetuned model')
    parser.add_argument('--model_finetuned_lora', type=str, default="", help='finetuned model with lora layer')
    parser.add_argument('--prompts', nargs='+', type=str, help='list of prompts')
    parser.add_argument('--num_images', type=int, default=30, help='number of images')
    parser.add_argument('--output_dir', type=str, default="/scratch/mp5847/diffusers_ckpt/output", help='output directory')
    parser.add_argument('--lora_edit_alpha', type=float, default=-0.97, help='amount of edit to lora layer')
    parser.add_argument('--tv_edit_alpha', type=float, default=0.5, help='amount of edit to task vector layer')
    parser.add_argument('--create_grid', action='store_true', help='if set, create grid of images')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_pretrained = StableDiffusionPipeline.from_pretrained(args.model_pretrained, torch_dtype=torch.float16)
    pipe_pretrained.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating images ...")
    print("Edit prompt: ", args.prompts)

    if(args.model_finetuned != ""): #task vector edit
        print("Sampling from standard finetuning edited model")
        pipe_finetuned = StableDiffusionPipeline.from_pretrained(args.model_finetuned, torch_dtype=torch.float16)
        pipe_finetuned.to("cuda")

        #edit process
        unet_pretrained = pipe_pretrained.unet
        unet_finetuned = pipe_finetuned.unet

        #save model unet
        torch.save(unet_pretrained, "unet_pretrained.pt")
        torch.save(unet_finetuned, "unet_finetuned.pt")

        task_vector = TaskVector(pretrained_checkpoint="unet_pretrained.pt", 
                                finetuned_checkpoint="unet_finetuned.pt")

        neg_task_vector = -task_vector
        unet_edited = neg_task_vector.apply_to("unet_pretrained.pt", scaling_coef=args.tv_edit_alpha)
        pipe_pretrained.unet = unet_edited
        pipe_finetuned = pipe_pretrained

        os.remove("unet_pretrained.pt")
        os.remove("unet_finetuned.pt")

    elif(args.model_finetuned_lora != ""):
        print("Sampling from lora finetuning edited model")
        pipe_finetuned = StableDiffusionPipeline.from_pretrained(args.model_pretrained, torch_dtype=torch.float16)
        pipe_finetuned.unet.load_attn_procs(args.model_finetuned_lora)
        pipe_finetuned.to("cuda")

        #scale lora layer
        for name, param in pipe_finetuned.unet.named_parameters():
            if("_lora.up.weight" in name):
                with torch.no_grad():
                    print("Editing lora layer: ", name)
                    #flip the sign of the lora layer
                    param.copy_(torch.nn.Parameter(args.lora_edit_alpha * param))

    for p in args.prompts:
        for i in range(args.num_images):
            while(True):
                nsfw = save_image(pipe_finetuned, p, os.path.join(args.output_dir, f"{p}_{i}.png"))
            
                #check if nsfw is a list
                if isinstance(nsfw, list):
                    nsfw = nsfw[0]

                if not nsfw:
                    break
    
    if(args.create_grid):
        for p in args.prompts:
            concat_images_in_square_grid(args.output_dir, p, os.path.join(args.output_dir, f"grid {p}.png"))

    print("Done!")
    