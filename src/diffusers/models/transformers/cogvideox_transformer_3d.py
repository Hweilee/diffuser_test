# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_torch_version, logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.final_dropout = final_dropout
        self.ff_inner_dim = ff_inner_dim
        self.ff_bias = ff_bias
        self.attention_out_bias = attention_out_bias

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_frames: int = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


# norm_final is just nn.LayerNorm, all ops will be on channel dimension, so we dont have to care about frame dimension
# proj_out is also just along channel dimension
# norm_out is just linear and nn.LayerNorm, again only on channel dimension, so we dont care about frame dims
# same story with norm1, norm2 and ff
# patch embed layer just applies on channel dim too and condenses to [B, FHW, C]
# only attention layer seems to be actually doing anything with the frame dimension and so only location where FreeNoise needs to be applied
# Since it does not matter for norm1, norm2, ff, and they might create memory bottleneck, just use FreeNoise frame split on them too


@maybe_allow_in_graph
class FreeNoiseCogVideoXBlock(nn.Module):
    r"""
    FreeNoise block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        context_length: int = 16,
        context_stride: int = 4,
        weighting_scheme: str = "pyramid",
        prompt_interpolation_callback: Callable[[int, int, torch.Tensor, torch.Tensor], torch.Tensor] = None,
        prompt_pooling_callback: Callable[[List[torch.Tensor]], torch.Tensor] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.final_dropout = final_dropout
        self.ff_inner_dim = ff_inner_dim
        self.ff_bias = ff_bias
        self.attention_out_bias = attention_out_bias

        self.set_free_noise_properties(
            context_length, context_stride, weighting_scheme, prompt_interpolation_callback, prompt_pooling_callback
        )

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    # Copied from diffusers.models.attention.FreeNoiseTransformerBlock.set_free_noise_properties
    def set_free_noise_properties(
        self,
        context_length: int,
        context_stride: int,
        weighting_scheme: str = "pyramid",
        prompt_interpolation_callback: Callable[[int, int, torch.Tensor, torch.Tensor], torch.Tensor] = None,
        prompt_pooling_callback: Callable[[List[torch.Tensor]], torch.Tensor] = None,
    ) -> None:
        if prompt_interpolation_callback is None:
            raise ValueError("Must pass a callback to interpolate between prompt embeddings.")
        if prompt_pooling_callback is None:
            raise ValueError("Must pass a callback to pool prompt embeddings.")
        self.context_length = context_length
        self.context_stride = context_stride
        self.weighting_scheme = weighting_scheme
        self.prompt_interpolation_callback = prompt_interpolation_callback
        self.prompt_pooling_callback = prompt_pooling_callback

    # Copied from diffusers.models.attention.FreeNoiseTransformerBlock._get_frame_indices
    def _get_frame_indices(self, num_frames: int) -> List[Tuple[int, int]]:
        frame_indices = []
        for i in range(0, num_frames - self.context_length + 1, self.context_stride):
            window_start = i
            window_end = min(num_frames, i + self.context_length)
            frame_indices.append((window_start, window_end))
        return frame_indices

    # Copied from diffusers.models.attention.FreeNoiseTransformerBlock._get_frame_weights
    def _get_frame_weights(self, num_frames: int, weighting_scheme: str = "pyramid") -> List[float]:
        if weighting_scheme == "flat":
            weights = [1.0] * num_frames

        elif weighting_scheme == "pyramid":
            if num_frames % 2 == 0:
                # num_frames = 4 => [1, 2, 2, 1]
                mid = num_frames // 2
                weights = list(range(1, mid + 1))
                weights = weights + weights[::-1]
            else:
                # num_frames = 5 => [1, 2, 3, 2, 1]
                mid = (num_frames + 1) // 2
                weights = list(range(1, mid))
                weights = weights + [mid] + weights[::-1]

        elif weighting_scheme == "delayed_reverse_sawtooth":
            if num_frames % 2 == 0:
                # num_frames = 4 => [0.01, 2, 2, 1]
                mid = num_frames // 2
                weights = [0.01] * (mid - 1) + [mid]
                weights = weights + list(range(mid, 0, -1))
            else:
                # num_frames = 5 => [0.01, 0.01, 3, 2, 1]
                mid = (num_frames + 1) // 2
                weights = [0.01] * mid
                weights = weights + list(range(mid, 0, -1))
        else:
            raise ValueError(f"Unsupported value for weighting_scheme={weighting_scheme}")

        return weights

    def _prepare_free_noise_encoder_hidden_states(
        self,
        encoder_hidden_states: Union[
            torch.Tensor, List[torch.Tensor], Tuple[Dict[int, torch.Tensor], Optional[Dict[int, torch.Tensor]]]
        ],
        frame_indices: List[int],
    ) -> List[torch.Tensor]:
        if torch.is_tensor(encoder_hidden_states):
            encoder_hidden_states = [encoder_hidden_states.clone() for _ in range(len(frame_indices))]

        elif isinstance(encoder_hidden_states, tuple):
            print("frame_indices:", frame_indices)
            pooled_prompt_embeds_list = []
            pooled_negative_prompt_embeds_list = []
            negative_prompt_embeds_dict, prompt_embeds_dict = encoder_hidden_states
            last_frame_start = 0

            # For every batch of frames that is to be processed, pool the positive and negative prompt embeddings.
            # TODO(aryan): Since this is experimental, I didn't try many different things. I found from testing
            # that pooling with previous batch frame embeddings necessary to produce better results and help with
            # prompt transitions.
            for frame_start, frame_end in frame_indices:
                pooled_prompt_embeds = None
                pooled_negative_prompt_embeds = None

                pooling_list = [
                    prompt_embeds_dict[i] for i in range(last_frame_start, frame_end) if i in prompt_embeds_dict
                ]
                if len(pooling_list) > 0:
                    print("pooling", [i for i in range(last_frame_start, frame_end) if i in prompt_embeds_dict])
                    pooled_prompt_embeds = self.prompt_pooling_callback(pooling_list)
                    print("after pooling:", pooled_prompt_embeds.isnan().any())

                if negative_prompt_embeds_dict is not None:
                    pooling_list = [
                        negative_prompt_embeds_dict[i]
                        for i in range(last_frame_start, frame_end)
                        if i in negative_prompt_embeds_dict
                    ]
                    if len(pooling_list) > 0:
                        print(
                            "negative pooling", [i for i in range(last_frame_start, frame_end) if i in prompt_embeds_dict]
                        )
                        pooled_negative_prompt_embeds = self.prompt_pooling_callback(pooling_list)
                        print("after negative pooling:", pooled_negative_prompt_embeds.isnan().any())

                pooled_prompt_embeds_list.append(pooled_prompt_embeds)
                pooled_negative_prompt_embeds_list.append(pooled_negative_prompt_embeds)
                last_frame_start = frame_start

            assert pooled_prompt_embeds_list[0] is not None
            assert pooled_prompt_embeds[-1] is not None
            if negative_prompt_embeds_dict is not None:
                assert pooled_negative_prompt_embeds_list[0] is not None
                assert pooled_negative_prompt_embeds_list[-1] is not None

            # If there were no relevant prompts for certain frame batches, interpolate and fill in the gaps
            last_existent_embed_index = 0
            for i in range(1, len(frame_indices)):
                if pooled_prompt_embeds_list[i] is not None and i - last_existent_embed_index > 1:
                    print("interpolating:", last_existent_embed_index, i)
                    interpolated_embeds = self.prompt_interpolation_callback(
                        last_existent_embed_index,
                        i,
                        pooled_prompt_embeds_list[last_existent_embed_index],
                        pooled_prompt_embeds_list[i],
                    )
                    print("after interpolating", interpolated_embeds.isnan().any())
                    pooled_prompt_embeds_list[last_existent_embed_index : i + 1] = interpolated_embeds.split(1, dim=0)
                    last_existent_embed_index = i
            assert all(x is not None for x in pooled_prompt_embeds_list)

            if negative_prompt_embeds_dict is not None:
                last_existent_embed_index = 0
                for i in range(1, len(frame_indices)):
                    if pooled_negative_prompt_embeds_list[i] is not None and i - last_existent_embed_index > 1:
                        print("negative interpolating:", last_existent_embed_index, i)
                        interpolated_embeds = self.prompt_interpolation_callback(
                            last_existent_embed_index,
                            i,
                            pooled_negative_prompt_embeds_list[last_existent_embed_index],
                            pooled_negative_prompt_embeds_list[i],
                        )
                        print("after negative interpolating", interpolated_embeds.isnan().any())
                        pooled_negative_prompt_embeds_list[
                            last_existent_embed_index : i + 1
                        ] = interpolated_embeds.split(1, dim=0)
                        last_existent_embed_index = i
                assert all(x is not None for x in pooled_negative_prompt_embeds_list)

            if negative_prompt_embeds_dict is not None:
                # Classifier-Free Guidance
                pooled_prompt_embeds_list = [
                    torch.cat([negative_prompt_embeds, prompt_embeds])
                    for negative_prompt_embeds, prompt_embeds in zip(
                        pooled_negative_prompt_embeds_list, pooled_prompt_embeds_list
                    )
                ]

            encoder_hidden_states = pooled_prompt_embeds_list

        elif not isinstance(encoder_hidden_states, list):
            raise ValueError(
                f"Expected `encoder_hidden_states` to be a tensor, list of tensor, or a tuple of dictionaries, but found {type(encoder_hidden_states)=}"
            )

        assert isinstance(encoder_hidden_states, list) and len(encoder_hidden_states) == len(frame_indices)
        return encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_frames: int = None,
    ) -> torch.Tensor:
        # hidden_states: [B, F x H x W, C]
        device = hidden_states.device
        dtype = hidden_states.dtype

        frame_indices = self._get_frame_indices(num_frames)
        frame_weights = self._get_frame_weights(self.context_length, self.weighting_scheme)
        frame_weights = (
            torch.tensor(frame_weights, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        is_last_frame_batch_complete = frame_indices[-1][1] == num_frames

        # Handle out-of-bounds case if num_frames isn't perfectly divisible by context_length
        # For example, num_frames=25, context_length=16, context_stride=4, then we expect the ranges:
        #    [(0, 16), (4, 20), (8, 24), (10, 26)]
        if not is_last_frame_batch_complete:
            if num_frames < self.context_length:
                raise ValueError(f"Expected {num_frames=} to be greater or equal than {self.context_length=}")
            last_frame_batch_length = num_frames - frame_indices[-1][1]
            frame_indices.append((num_frames - self.context_length, num_frames))

        # Unflatten frame dimension: [B, F, HW, C]
        batch_size, frames_height_width, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, num_frames, frames_height_width // num_frames, channels)
        encoder_hidden_states = self._prepare_free_noise_encoder_hidden_states(encoder_hidden_states, frame_indices)

        num_times_accumulated = torch.zeros((1, num_frames, 1, 1), device=device)
        accumulated_values = torch.zeros_like(hidden_states)

        text_seq_length = _get_text_seq_length(encoder_hidden_states)

        for i, (frame_start, frame_end) in enumerate(frame_indices):
            # The reason for slicing here is to handle cases like frame_indices=[(0, 16), (16, 20)],
            # if the user provided a video with 19 frames, or essentially a non-multiple of `context_length`.
            weights = torch.ones_like(num_times_accumulated[:, frame_start:frame_end])
            weights *= frame_weights

            # Flatten frame dimension: [B, F'HW, C]
            hidden_states_chunk = hidden_states[:, frame_start:frame_end].flatten(1, 2)
            print(
                "debug:",
                text_seq_length,
                torch.isnan(hidden_states_chunk).any(),
                torch.isnan(encoder_hidden_states[i]).any(),
            )

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states_chunk, encoder_hidden_states[i], temb
            )

            # attention
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

            hidden_states_chunk = hidden_states_chunk + gate_msa * attn_hidden_states
            encoder_hidden_states[i] = encoder_hidden_states[i] + enc_gate_msa * attn_encoder_hidden_states

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                hidden_states_chunk, encoder_hidden_states[i], temb
            )

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states_chunk = hidden_states_chunk + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states[i] = encoder_hidden_states[i] + enc_gate_ff * ff_output[:, :text_seq_length]

            # Unflatten frame dimension: [B, F', HW, C]
            _num_frames = frame_end - frame_start
            hidden_states_chunk = hidden_states_chunk.reshape(batch_size, _num_frames, -1, channels)

            if i == len(frame_indices) - 1 and not is_last_frame_batch_complete:
                accumulated_values[:, -last_frame_batch_length:] += (
                    hidden_states_chunk[:, -last_frame_batch_length:] * weights[:, -last_frame_batch_length:]
                )
                num_times_accumulated[:, -last_frame_batch_length:] += weights[:, -last_frame_batch_length]
            else:
                accumulated_values[:, frame_start:frame_end] += hidden_states_chunk * weights
                num_times_accumulated[:, frame_start:frame_end] += weights

        # TODO(aryan): Maybe this could be done in a better way.
        #
        # Previously, this was:
        # hidden_states = torch.where(
        #    num_times_accumulated > 0, accumulated_values / num_times_accumulated, accumulated_values
        # )
        #
        # The reasoning for the change here is `torch.where` became a bottleneck at some point when golfing memory
        # spikes. It is particularly noticeable when the number of frames is high. My understanding is that this comes
        # from tensors being copied - which is why we resort to spliting and concatenating here. I've not particularly
        # looked into this deeply because other memory optimizations led to more pronounced reductions.
        hidden_states = torch.cat(
            [
                torch.where(num_times_split > 0, accumulated_split / num_times_split, accumulated_split)
                for accumulated_split, num_times_split in zip(
                    accumulated_values.split(self.context_length, dim=1),
                    num_times_accumulated.split(self.context_length, dim=1),
                )
            ],
            dim=1,
        ).to(dtype)

        # Flatten frame dimension: [B, FHW, C]
        hidden_states = hidden_states.flatten(1, 2)

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        text_seq_length = _get_text_seq_length(encoder_hidden_states)
        encoder_hidden_states, hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    num_frames,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    num_frames=num_frames,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def _get_text_seq_length(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.shape[1]
    if isinstance(x, list):
        return _get_text_seq_length(x[0])
    if isinstance(x, dict):
        return _get_text_seq_length(next(iter(x.values())))
    return None
