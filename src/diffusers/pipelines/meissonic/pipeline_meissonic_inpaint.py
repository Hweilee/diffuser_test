# Copyright 2024 The HuggingFace Team and The MeissonFlow Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import VQModel
from ...models.transformers import MeissonicTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import MeissonicScheduler
from ...utils import replace_example_docstring

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> pipe(prompt, input_image, mask).images[0].save("out.png")
        ```
"""


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    # latent_image_ids = latent_image_ids.unsqueeze(0).repeat(batch_size, 1, 1)

    return latent_image_ids.to(device=device, dtype=dtype)


class MeissonicInpaintPipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: MeissonicTransformer2DModel
    scheduler: MeissonicScheduler

    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    # TODO - when calling self.vqvae.quantize, it uses self.vqvae.quantize.embedding.weight before
    # the forward method of self.vqvae.quantize, so the hook doesn't get called to move the parameter
    # off the meta device. There should be a way to fix this instead of just not offloading it
    _exclude_from_cpu_offload = ["vqvae"]

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: MeissonicTransformer2DModel,
        scheduler: MeissonicScheduler,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
            do_resize=True,
        )
        self.scheduler.register_to_config(masking_schedule="linear")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        strength: float = 1.0,
        num_inference_steps: int = 12,
        guidance_scale: float = 10.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        output_type="pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        micro_conditioning_aesthetic_score: int = 6,
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 16):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                Analogous to `encoder_hidden_states` for the positive prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                The targeted aesthetic score according to the laion aesthetic classifier. See
                https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                The targeted height, width crop coordinates. See the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                Configures the temperature scheduler on `self.scheduler` see `MeissonicScheduler#set_timesteps`.

        Examples:

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """

        if (prompt_embeds is not None and encoder_hidden_states is None) or (
            prompt_embeds is None and encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `prompt_embeds` and `encoder_hidden_states` or neither"
            )

        if (
            negative_prompt_embeds is not None
            and negative_encoder_hidden_states is None
        ) or (
            negative_prompt_embeds is None
            and negative_encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
            )

        if (prompt is None and prompt_embeds is None) or (
            prompt is not None and prompt_embeds is not None
        ):
            raise ValueError("pass only one of `prompt` or `prompt_embeds`")

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        if prompt_embeds is None:
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,  # self.tokenizer.model_max_length,
            ).input_ids.to(self._execution_device)

            outputs = self.text_encoder(
                input_ids, return_dict=True, output_hidden_states=True
            )
            prompt_embeds = outputs.text_embeds
            encoder_hidden_states = outputs.hidden_states[-2]

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(
            num_images_per_prompt, 1, 1
        )

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]

                input_ids = self.tokenizer(
                    negative_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77,  # self.tokenizer.model_max_length,
                ).input_ids.to(self._execution_device)

                outputs = self.text_encoder(
                    input_ids, return_dict=True, output_hidden_states=True
                )
                negative_prompt_embeds = outputs.text_embeds
                negative_encoder_hidden_states = outputs.hidden_states[-2]

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                num_images_per_prompt, 1
            )
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(
                num_images_per_prompt, 1, 1
            )

            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat(
                [negative_encoder_hidden_states, encoder_hidden_states]
            )

        image = self.image_processor.preprocess(image)

        height, width = image.shape[-2:]

        # Note that the micro conditionings _do_ flip the order of width, height for the original size
        # and the crop coordinates. This is how it was done in the original code base
        micro_conds = torch.tensor(
            [
                width,
                height,
                micro_conditioning_crop_coord[0],
                micro_conditioning_crop_coord[1],
                micro_conditioning_aesthetic_score,
            ],
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )

        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(
            2 * batch_size if guidance_scale > 1.0 else batch_size, -1
        )

        self.scheduler.set_timesteps(
            num_inference_steps, temperature, self._execution_device
        )
        num_inference_steps = int(len(self.scheduler.timesteps) * strength)
        start_timestep_idx = len(self.scheduler.timesteps) - num_inference_steps

        needs_upcasting = False  # self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast

        if needs_upcasting:
            self.vqvae.float()

        latents = self.vqvae.encode(
            image.to(dtype=self.vqvae.dtype, device=self._execution_device)
        ).latents
        latents_bsz, channels, latents_height, latents_width = latents.shape
        latents = self.vqvae.quantize(latents)[2][2].reshape(
            latents_bsz, latents_height, latents_width
        )

        mask = self.mask_processor.preprocess(
            mask_image, height // self.vae_scale_factor, width // self.vae_scale_factor
        )
        mask = (
            mask.reshape(mask.shape[0], latents_height, latents_width)
            .bool()
            .to(latents.device)
        )
        latents[mask] = self.scheduler.config.mask_token_id

        starting_mask_ratio = mask.sum() / latents.numel()

        latents = latents.repeat(num_images_per_prompt, 1, 1)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(start_timestep_idx, len(self.scheduler.timesteps)):
                timestep = self.scheduler.timesteps[i]

                if guidance_scale > 1.0:
                    model_input = torch.cat([latents] * 2)
                else:
                    model_input = latents

                if height == 1024:  # args.resolution == 1024:
                    img_ids = _prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[-2],
                        model_input.shape[-1],
                        model_input.device,
                        model_input.dtype,
                    )
                else:
                    img_ids = _prepare_latent_image_ids(
                        model_input.shape[0],
                        2 * model_input.shape[-2],
                        2 * model_input.shape[-1],
                        model_input.device,
                        model_input.dtype,
                    )
                txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                )
                model_output = self.transformer(
                    model_input,
                    micro_conds=micro_conds,
                    pooled_projections=prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    # cross_attention_kwargs=cross_attention_kwargs,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    timestep=torch.tensor(
                        [timestep], device=model_input.device, dtype=torch.long
                    ),
                )

                if guidance_scale > 1.0:
                    uncond_logits, cond_logits = model_output.chunk(2)
                    model_output = uncond_logits + guidance_scale * (
                        cond_logits - uncond_logits
                    )

                latents = self.scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                    starting_mask_ratio=starting_mask_ratio,
                ).prev_sample

                if i == len(self.scheduler.timesteps) - 1 or (
                    (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, timestep, latents)

        if output_type == "latent":
            output = latents
        else:
            output = self.vqvae.decode(
                latents,
                force_not_quantize=True,
                shape=(
                    batch_size,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                    self.vqvae.config.latent_channels,
                ),
            ).sample.clip(0, 1)
            output = self.image_processor.postprocess(output, output_type)

            if needs_upcasting:
                self.vqvae.half()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (output,)

        return ImagePipelineOutput(output)
