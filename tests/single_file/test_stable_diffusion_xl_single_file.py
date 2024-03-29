import gc
import unittest

import torch

from diffusers import (
    DDIMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLPipelineSingleFileSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_file_format_inference_is_same(self):
        ckpt_path = (
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
        )

        sf_pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path)
        sf_pipe.scheduler = DDIMScheduler.from_config(sf_pipe.scheduler.config)
        sf_pipe.unet.set_attn_processor(AttnProcessor())
        sf_pipe.to("cuda")

        generator = torch.Generator(device="cpu").manual_seed(0)
        image_single_file = sf_pipe("a turtle", num_inference_steps=2, generator=generator, output_type="np").images[0]

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_attn_processor(AttnProcessor())
        pipe.to("cuda")

        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe("a turtle", num_inference_steps=2, generator=generator, output_type="np").images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < 1e-3

    def test_single_file_component_configs(self):
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

        ckpt_path = (
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
        )
        single_file_pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, load_safety_checker=True)

        for param_name, param_value in single_file_pipe.text_encoder.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder.config.to_dict()[param_name] == param_value

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "architectures", "_use_default_values"]
        for param_name, param_value in single_file_pipe.unet.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.unet.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

        for param_name, param_value in single_file_pipe.vae.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.vae.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

        for param_name, param_value in single_file_pipe.safety_checker.config.to_dict().items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.safety_checker.config.to_dict()[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"
