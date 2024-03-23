# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .autoencoders import AutoencoderKL
from .embeddings import (
    TimestepEmbedding,
)
from .modeling_utils import ModelMixin
from .unets.unet_2d_blocks import Downsample2D, ResnetBlock2D, Transformer2DModel, UNetMidBlock2DCrossAttn, Upsample2D
from .unets.unet_2d_condition import UNet2DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetXSOutput(BaseOutput):
    """
    The output of [`UNetControlNetXSModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The output of the `UNetControlNetXSModel`. Unlike `ControlNetOutput` this is NOT to be added to the base model
            output, but is already the final output.
    """

    sample: torch.FloatTensor = None


# copied from diffusers.models.controlnet.ControlNetConditioningEmbedding
class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetXSAddon(ModelMixin, ConfigMixin):
    r"""
    A `ControlNetXSAddon` model. To use it, pass it into a `ControlNetXSModel` (together with a `UNet2DConditionModel` base model).

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    Like `ControlNetXSModel`, `ControlNetXSAddon` is compatible with StableDiffusion and StableDiffusion-XL.
    It's default parameters are compatible with StableDiffusion.

    Parameters:
        conditioning_channels (`int`, defaults to 3):
            Number of channels of conditioning input (e.g. an image)
        conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
            The tuple of output channels for each block in the `controlnet_cond_embedding` layer.
        time_embedding_input_dim (`int`, defaults to 320):
            Dimension of input into time embedding. Needs to be same as in the base model.
        time_embedding_dim (`int`, defaults to 1280):
            Dimension of output from time embedding. Needs to be same as in the base model.
        time_embedding_mix (`float`, defaults to 1.0):
            If 0, then only the control addon's time embedding is used.
            If 1, then only the base unet's time embedding is used.
            Otherwise, both are combined.
        learn_time_embedding (`bool`, defaults to `False`):
            Whether a time embedding should be learned. If yes, `ControlNetXSModel` will combine the time embeddings of the base model and the addon.
            If no, `ControlNetXSModel` will use the base model's time embedding.
        channels_base (`Dict[str, List[Tuple[int]]]`, defaults to `ControlNetXSAddon.gather_base_subblock_sizes((320,640,1280,1280))`):
            Channels of each subblock of the base model. Use `ControlNetXSAddon.gather_base_subblock_sizes` to obtain them.
        attention_head_dim (`list[int]`, defaults to `[4]`):
            The dimension of the attention heads.
        block_out_channels (`list[int]`, defaults to `[4, 8, 16, 16]`):
            The tuple of output channels for each block.
        cross_attention_dim (`int`, defaults to 1024):
            The dimension of the cross attention features.
        down_block_types (`list[str]`, defaults to `["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]`):
            The tuple of downsample blocks to use.
        sample_size (`int`, defaults to 96):
            Height and width of input/output sample.
        transformer_layers_per_block (`Union[int, Tuple[int]]`, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        upcast_attention (`bool`, defaults to `True`):
            Whether the attention computation should always be upcasted.
        max_norm_num_groups (`int`, defaults to 32):
            Maximum number of groups in group normal. The actual number will the the largest divisor of the respective channels, that is <= max_norm_num_groups.
    """

    @staticmethod
    def gather_base_subblock_sizes(blocks_sizes: List[int]):
        """
        To create a correctly sized `ControlNetXSAddon`, we need to know
        the channels sizes of each base subblock.

        Parameters:
            blocks_sizes (`List[int]`):
                Channel sizes of each base block.
        """

        n_blocks = len(blocks_sizes)
        n_subblocks_per_block = 3

        down_out = []
        up_in = []

        # down_out
        for b in range(n_blocks):
            for i in range(n_subblocks_per_block):
                if b == n_blocks - 1 and i == 2:
                    # Last block has no downsampler, so there are only 2 subblocks instead of 3
                    continue

                # The input channels are changed by the first resnet, which is in the first subblock.
                if i == 0:
                    # Same input channels
                    down_out.append(blocks_sizes[max(b - 1, 0)])
                else:
                    # Changed input channels
                    down_out.append(blocks_sizes[b])

        down_out.append(blocks_sizes[-1])

        # up_in
        rev_blocks_sizes = list(reversed(blocks_sizes))
        for b in range(len(rev_blocks_sizes)):
            for i in range(n_subblocks_per_block):
                # The input channels are changed by the first resnet, which is in the first subblock.
                if i == 0:
                    # Same input channels
                    up_in.append(rev_blocks_sizes[max(b - 1, 0)])
                else:
                    # Changed input channels
                    up_in.append(rev_blocks_sizes[b])

        return {
            "down - out": down_out,
            "mid - out": blocks_sizes[-1],
            "up - in": up_in,
        }

    @classmethod
    def from_unet(
        cls,
        base_model: UNet2DConditionModel,
        size_ratio: Optional[float] = None,
        block_out_channels: Optional[List[int]] = None,
        num_attention_heads: Optional[List[int]] = None,
        learn_time_embedding: bool = False,
        time_embedding_mix: int = 1.0,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        r"""
        Instantiate a [`ControlNetXSAddon`] from a [`UNet2DConditionModel`].

        Parameters:
            base_model (`UNet2DConditionModel`):
                The UNet model we want to control. The dimensions of the ControlNetXSAddon will be adapted to it.
            size_ratio (float, *optional*, defaults to `None`):
                When given, block_out_channels is set to a fraction of the base model's block_out_channels.
                Either this or `block_out_channels` must be given.
            block_out_channels (`List[int]`, *optional*, defaults to `None`):
                Down blocks output channels in control model. Either this or `size_ratio` must be given.
            num_attention_heads (`List[int]`, *optional*, defaults to `None`):
                The dimension of the attention heads. The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
            learn_time_embedding (`bool`, defaults to `False`):
                Whether the `ControlNetXSAddon` should learn a time embedding.
            conditioning_embedding_out_channels (`Tuple[int]`, defaults to `(16, 32, 96, 256)`):
                The tuple of output channel for each block in the `controlnet_cond_embedding` layer.
        """

        # Check input
        fixed_size = block_out_channels is not None
        relative_size = size_ratio is not None
        if not (fixed_size ^ relative_size):
            raise ValueError(
                "Pass exactly one of `block_out_channels` (for absolute sizing) or `size_ratio` (for relative sizing)."
            )

        channels_base = ControlNetXSAddon.gather_base_subblock_sizes(base_model.config.block_out_channels)

        block_out_channels = [int(b * size_ratio) for b in base_model.config.block_out_channels]
        if num_attention_heads is None:
            # The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
            num_attention_heads = base_model.config.attention_head_dim

        max_norm_num_groups = base_model.config.norm_num_groups

        time_embedding_input_dim = base_model.time_embedding.linear_1.in_features
        time_embedding_dim = base_model.time_embedding.linear_1.out_features

        return ControlNetXSAddon(
            learn_time_embedding=learn_time_embedding,
            channels_base=channels_base,
            attention_head_dim=num_attention_heads,
            block_out_channels=block_out_channels,
            cross_attention_dim=base_model.config.cross_attention_dim,
            down_block_types=base_model.config.down_block_types,
            sample_size=base_model.config.sample_size,
            transformer_layers_per_block=base_model.config.transformer_layers_per_block,
            upcast_attention=base_model.config.upcast_attention,
            max_norm_num_groups=max_norm_num_groups,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            time_embedding_input_dim=time_embedding_input_dim,
            time_embedding_dim=time_embedding_dim,
            time_embedding_mix=time_embedding_mix,
        )

    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        time_embedding_input_dim: Optional[int] = 320,
        time_embedding_dim: Optional[int] = 1280,
        time_embedding_mix: float = 1.0,
        learn_time_embedding: bool = False,
        channels_base: Dict[str, List[Tuple[int]]] = {
            "down - out": [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280],
            "mid - out": 1280,
            "up - in": [1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320],
        },
        attention_head_dim: Union[int, Tuple[int]] = 4,
        block_out_channels: Tuple[int] = (4, 8, 16, 16),
        cross_attention_dim: int = 1024,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        sample_size: Optional[int] = 96,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        upcast_attention: bool = True,
        max_norm_num_groups: int = 32,
    ):
        super().__init__()

        self.sample_size = sample_size

        # `num_attention_heads` defaults to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = attention_head_dim

        # Check inputs
        if conditioning_channel_order not in ["rgb", "bgr"]:
            raise ValueError(f"unknown `conditioning_channel_order`: {conditioning_channel_order}")

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )
        elif isinstance(attention_head_dim, int):
            attention_head_dim = [attention_head_dim] * len(down_block_types)

        # input
        self.conv_in = nn.Conv2d(4, block_out_channels[0], kernel_size=3, padding=1)

        # time
        if learn_time_embedding:
            self.time_embedding = TimestepEmbedding(time_embedding_input_dim, time_embedding_dim)
        else:
            self.time_embedding = None

        self.time_embed_act = None

        self.down_subblocks = nn.ModuleList([])
        self.up_subblocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        subblock_counter = 0

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            use_crossattention = down_block_type == "CrossAttnDownBlock2D"

            self.down_subblocks.append(
                CrossAttnDownSubBlock2D(
                    has_crossattn=use_crossattention,
                    in_channels=input_channel + channels_base["down - out"][subblock_counter],
                    out_channels=output_channel,
                    temb_channels=time_embedding_dim,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    max_norm_num_groups=max_norm_num_groups,
                )
            )
            subblock_counter += 1
            self.down_subblocks.append(
                CrossAttnDownSubBlock2D(
                    has_crossattn=use_crossattention,
                    in_channels=output_channel + channels_base["down - out"][subblock_counter],
                    out_channels=output_channel,
                    temb_channels=time_embedding_dim,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    max_norm_num_groups=max_norm_num_groups,
                )
            )
            subblock_counter += 1
            if i < len(down_block_types) - 1:
                self.down_subblocks.append(
                    DownSubBlock2D(
                        in_channels=output_channel + channels_base["down - out"][subblock_counter],
                        out_channels=output_channel,
                    )
                )
                subblock_counter += 1

        # mid
        mid_in_channels = block_out_channels[-1] + channels_base["down - out"][subblock_counter]
        mid_out_channels = block_out_channels[-1]

        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_in_channels,
            out_channels=mid_out_channels,
            temb_channels=time_embedding_dim,
            resnet_eps=1e-05,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=find_largest_factor(mid_in_channels, max_norm_num_groups),
            resnet_groups_out=find_largest_factor(mid_out_channels, max_norm_num_groups),
            use_linear_projection=True,
            upcast_attention=upcast_attention,
        )

        # 3 - Gather Channel Sizes
        channels_ctrl = {
            "down - out": [self.conv_in.out_channels] + [s.out_channels for s in self.down_subblocks],
            "mid - out": self.down_subblocks[-1].out_channels,
        }

        # 4 - Build connections between base and control model
        # b2c = base -> ctrl ; c2b = ctrl -> base
        self.down_zero_convs_b2c = nn.ModuleList([])
        self.down_zero_convs_c2b = nn.ModuleList([])
        self.mid_zero_convs_c2b = nn.ModuleList([])
        self.up_zero_convs_c2b = nn.ModuleList([])

        # 4.1 - Connections from base encoder to ctrl encoder
        # As the information is concatted to ctrl, the channels sizes don't change.
        for c in channels_base["down - out"]:
            self.down_zero_convs_b2c.append(make_zero_conv(c, c))

        # 4.2 - Connections from ctrl encoder to base encoder
        # As the information is added to base, the out-channels need to match base.
        for ch_base, ch_ctrl in zip(channels_base["down - out"], channels_ctrl["down - out"]):
            self.down_zero_convs_c2b.append(make_zero_conv(ch_ctrl, ch_base))

        # 4.3 - Connections in mid block
        self.mid_zero_convs_c2b = make_zero_conv(channels_ctrl["mid - out"], channels_base["mid - out"])

        # 4.3 - Connections from ctrl encoder to base decoder
        skip_channels = reversed(channels_ctrl["down - out"])
        for s, i in zip(skip_channels, channels_base["up - in"]):
            self.up_zero_convs_c2b.append(make_zero_conv(s, i))

        # 5 - Create conditioning hint embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

    def forward(self, *args, **kwargs):
        raise ValueError(
            "A ControlNetXSAddonModel cannot be run by itself. Pass it into a ControlNetXSModel model instead."
        )


class UNetControlNetXSModel(ModelMixin, ConfigMixin):
    r"""
    A UNet fused with a ControlNet-XS addon model

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    `UNetControlNetXSModel` is compatible with StableDiffusion and StableDiffusion-XL.
    It's default parameters are compatible with StableDiffusion.

    Most of it's paremeters are passed to the underlying `UNet2DConditionModel`. See it's documentation for details.

    Parameters:
        time_embedding_mix (`float`, defaults to 1.0):
            If 0, then only the control addon's time embedding is used.
            If 1, then only the base unet's time embedding is used.
            Otherwise, both are combined.
        ctrl_conditioning_channels (`int`, defaults to 3):
            The number of channels of the control conditioning input.
        ctrl_conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
            Block sizes of the `ControlNetConditioningEmbedding`.
        ctrl_conditioning_channel_order (`str`, defaults to "rgb"):
            The order of channels in the control conditioning input.
        ctrl_learn_time_embedding (`bool`, defaults to False):
            Whether the control addon should learn a time embedding. Needs to be `True` if `time_embedding_mix` > 0.
        ctrl_block_out_channels (`tuple[int]`, defaults to `(4, 8, 16, 16)`):
            The tuple of output channels for each block in the control addon.
        ctrl_attention_head_dim (`int` or `tuple[int]`, defaults to 4):
            The dimension of the attention heads in the control addon.
        ctrl_max_norm_num_groups (`int`, defaults to 32):
            The maximum number of groups to use for the normalization in the control addon. Can be reduced to fit the block sizes.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # unet configs
        sample_size: Optional[int] = 96,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = 8,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        upcast_attention: bool = True,
        time_embedding_dim: Optional[int] = None,
        time_cond_proj_dim: Optional[int] = None,
        projection_class_embeddings_input_dim: Optional[int] = None,
        # additional controlnet configs
        time_embedding_mix: float = 1.0,
        ctrl_conditioning_channels: int = 3,
        ctrl_conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        ctrl_conditioning_channel_order: str = "rgb",
        ctrl_learn_time_embedding: bool = False,
        ctrl_block_out_channels: Tuple[int] = (4, 8, 16, 16),
        ctrl_attention_head_dim: Union[int, Tuple[int]] = 4,
        ctrl_max_norm_num_groups: int = 32,
    ):
        super().__init__()

        if time_embedding_mix < 0 or time_embedding_mix > 1:
            raise ValueError("`time_embedding_mix` needs to be between 0 and 1.")
        if time_embedding_mix < 1 and not ctrl_learn_time_embedding:
            raise ValueError(
                "To use `time_embedding_mix` < 1, initialize `ctrl_addon` with `learn_time_embedding = True`"
            )

        def repeat_if_not_list(value, repetitions):
            return value if isinstance(value, (tuple, list)) else [value] * repetitions

        transformer_layers_per_block = repeat_if_not_list(transformer_layers_per_block, repetitions=len(down_block_types))
        cross_attention_dim = repeat_if_not_list(cross_attention_dim, repetitions=len(down_block_types))
        num_attention_heads = repeat_if_not_list(num_attention_heads, repetitions=len(down_block_types))

        time_embedding_dim = time_embedding_dim or block_out_channels[0] * 4

        # Create UNet and decompose it into subblocks, which we then save
        base_model = UNet2DConditionModel(
            sample_size=sample_size,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            attention_head_dim=num_attention_heads,
            use_linear_projection=True,
            upcast_attention=upcast_attention,
            time_embedding_dim=time_embedding_dim,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            time_cond_proj_dim=time_cond_proj_dim,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            addition_time_embed_dim=addition_time_embed_dim,
        )

        self.in_channels = 4

        self.base_time_proj = base_model.time_proj
        self.base_time_embedding = base_model.time_embedding
        self.base_class_embedding = base_model.class_embedding
        self.base_add_time_proj = base_model.add_time_proj if hasattr(base_model, "add_time_proj") else None
        self.base_add_embedding = base_model.add_embedding if hasattr(base_model, "add_embedding") else None

        self.base_conv_in = base_model.conv_in
        self.base_mid_block = base_model.mid_block
        self.base_conv_norm_out = base_model.conv_norm_out
        self.base_conv_act = base_model.conv_act
        self.base_conv_out = base_model.conv_out

        down_blocks = []
        up_blocks = []

        # create down blocks
        def left_shifted_iterator_pairs(iterable, keys=["in", "out"]):
            """e.g. [0,1,2,3] -> [({"in":0,"out":0}, {"in":0,"out":1}, {"in":1,"out":2}, {"in":2,"out":3}]"""
            left_shifted_iterable = iterable[0] + list(iterable[:-1])
            return [
                {keys[0]: o1, keys[1]: o2}
                for o1,o2 in zip(left_shifted_iterable, iterable)
            ]

        channels = {"base": left_shifted_iterator_pairs(block_out_channels), "ctrl": left_shifted_iterator_pairs(ctrl_block_out_channels)}

        for i, (down_block_type, b_channels, c_channels) in enumerate((down_block_types, channels["base"], channels["ctrl"])):
            has_crossattn = "CrossAttn" in down_block_type
            add_downsample = i==len(down_block_types)-1

            down_blocks.append(ControlNetXSCrossAttnDownBlock2D(
                base_in_channels = b_channels["in"],
                base_out_channels = b_channels["out"],
                ctrl_in_channels = c_channels["in"],
                ctrl_out_channels = c_channels["out"],
                temb_channels = base_model.config.time_embedding_dim,
                max_norm_num_groups = ctrl_max_norm_num_groups.max_norm_num_groups,
                has_crossattn = has_crossattn,
                transformer_layers_per_block = transformer_layers_per_block[i],
                num_attention_heads = num_attention_heads[i],
                cross_attention_dim = cross_attention_dim[i],
                add_downsample = add_downsample,
                upcast_attention = upcast_attention
            ))

        # create down blocks
        self.mid_block = ControlNetXSCrossAttnMidBlock2D(
                base_channels=block_out_channels[-1],
                ctrl_channels=ctrl_block_out_channels[-1],
                temb_channels = base_model.config.time_embedding_dim,
                transformer_layers_per_block = transformer_layers_per_block[-1],
                num_attention_heads = num_attention_heads[-1],
                cross_attention_dim = cross_attention_dim[-1],
                upcast_attention = upcast_attention,
        )

        # create up blocks
        rev_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        rev_num_attention_heads = list(reversed(num_attention_heads))
        rev_cross_attention_dim = list(reversed(cross_attention_dim))
        rev_block_out_channels = list(reversed(block_out_channels))

        for i, up_block_type in enumerate(up_block_types):
            has_crossattn = "CrossAttn" in down_block_type
            add_upsample = i>0  # todo umer: correct?

            up_blocks.append(ControlNetXSCrossAttnUpBlock2D(# todo umer
                in_channels = 123456,
                out_channels = 123456,
                prev_output_channel = 123456,
                ctrl_skip_channels = [123456, 123456],
                temb_channels = base_model.config.time_embedding_dim,
                has_crossattn = has_crossattn,
                transformer_layers_per_block = rev_transformer_layers_per_block[-1],
                num_attention_heads = rev_num_attention_heads[-1],
                cross_attention_dim = rev_cross_attention_dim[-1],
                add_upsample = add_upsample,
                upcast_attention = upcast_attention,
            ))

        self.down_bocks = nn.ModuleList(down_blocks)
        self.up_bocks = nn.ModuleList(up_blocks)

    @classmethod
    def from_unet2d(
        cls,
        unet: UNet2DConditionModel,
        controlnet: ControlNetXSAddon,
        load_weights: bool = True,
    ):
        # Create config for UNetControlNetXSModel object
        config = {}
        config["_class_name"] = cls.__name__

        params_for_unet = [
            "sample_size",
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "norm_num_groups",
            "cross_attention_dim",
            "transformer_layers_per_block",
            "class_embed_type",
            "addition_embed_type",
            "addition_time_embed_dim",
            "upcast_attention",
            "time_embedding_dim",
            "time_cond_proj_dim",
            "projection_class_embeddings_input_dim",
        ]
        config.update({k: v for k, v in unet.config.items() if k in params_for_unet})
        # The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
        config["num_attention_heads"] = unet.config.attention_head_dim

        params_for_controlnet = [
            "conditioning_channels",
            "conditioning_embedding_out_channels",
            "conditioning_channel_order",
            "learn_time_embedding",
            "block_out_channels",
            "attention_head_dim",
            "max_norm_num_groups",
        ]
        config.update({"ctrl_" + k: v for k, v in controlnet.config.items() if k in params_for_controlnet})

        model = cls.from_config(config)

        if not load_weights:
            return model

        # Load params
        modules_from_unet = [
            "time_proj",
            "time_embedding",
            "conv_in",
            "mid_block",
            "conv_norm_out",
            "conv_act",
            "conv_out",
        ]
        for m in modules_from_unet:
            getattr(model, "base_" + m).load_state_dict(getattr(unet, m).state_dict())

        optional_modules_from_unet = ["class_embedding"]
        for m in optional_modules_from_unet:
            module = getattr(model, "base_" + m)
            if module is not None:
                module.load_state_dict(getattr(unet, m).state_dict())

        sdxl_specific_modules_from_unet = [
            "add_time_proj",
            "add_embedding",
        ]
        if hasattr(unet, sdxl_specific_modules_from_unet[0]):
            # if the UNet has any of the sdxl-specific components, it is an sdxl and has all of them
            for m in sdxl_specific_modules_from_unet:
                getattr(model, "base_" + m).load_state_dict(getattr(unet, m).state_dict())

        model.base_down_subblocks, model.base_up_subblocks = UNetControlNetXSModel._unet_to_subblocks(unet)

        model.control_addon.load_state_dict(controlnet.state_dict())

        # ensure that the UNetControlNetXSModel is the same dtype as the UNet2DConditionModel
        model.to(unet.dtype)

        return model

    def freeze_unet2d_params(self) -> None:
        """Freeze the weights of just the UNet2DConditionModel, and leave the ControlNetXSAddon
        unfrozen for fine tuning.
        """
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze ControlNetXSAddon
        for param in self.control_addon.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def _check_if_vae_compatible(self, vae: AutoencoderKL):
        condition_downscale_factor = 2 ** (len(self.control_addon.config.conditioning_embedding_out_channels) - 1)
        vae_downscale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        compatible = condition_downscale_factor == vae_downscale_factor
        return compatible, condition_downscale_factor, vae_downscale_factor

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor] = None,
        conditioning_scale: Optional[float] = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
        do_control: bool = True,
    ) -> Union[ControlNetXSOutput, Tuple]:
        """
        The [`ControlNetXSModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.FloatTensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                How much the control model affects the base model outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.
            do_control (`bool`, defaults to `True`):
                If `False`, the input is run only through the base model.

        Returns:
            [`~models.controlnetxs.ControlNetXSOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnetxs.ControlNetXSOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        # check channel order
        if self.control_addon.config.conditioning_channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.base_time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.ctrl_learn_time_embedding:
            ctrl_temb = self.control_addon.time_embedding(t_emb, timestep_cond)
            base_temb = self.base_time_embedding(t_emb, timestep_cond)
            interpolation_param = self.control_addon.config.time_embedding_mix**0.3

            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = self.base_time_embedding(t_emb)

        # added time & text embeddings
        aug_emb = None

        if self.base_class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.base_time_proj(class_labels)

            class_emb = self.base_class_embedding(class_labels).to(dtype=self.dtype)
            temb = temb + class_emb

        if self.config.addition_embed_type is None:
            pass
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.base_add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(temb.dtype)
            aug_emb = self.base_add_embedding(add_embeds)
        else:
            raise ValueError(
                f"ControlNet-XS currently only supports StableDiffusion and StableDiffusion-XL, so addition_embed_type = {self.config.addition_embed_type} is currently not supported."
            )

        temb = temb + aug_emb if aug_emb is not None else temb

        # text embeddings
        cemb = encoder_hidden_states

        # Preparation
        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []

        # Cross Control
        # Let's first define variables to shorten notation

        guided_hint = self.control_addon.controlnet_cond_embedding(controlnet_cond)

        # 1 - conv in & down

        h_base = self.base_conv_in(h_base)
        h_ctrl = self.control_addon.conv_in(h_ctrl)
        if guided_hint is not None:
            h_ctrl += guided_hint
        h_base = h_base + self.pre_zero_convs_c2b(h_ctrl) * conditioning_scale  # add ctrl -> base # todo umer: define self.pre_zero_convs_c2b

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        for down in self.down_blocks: # todo umer: define self.down_blocks
            h_base,h_ctrl,residual_hb,residual_hc = down(h_base,h_ctrl, temb, cemb, attention_mask, cross_attention_kwargs)
            hs_base.extend(residual_hb)
            hs_ctrl.extend(residual_hc)

        # 2 - mid
        h_base,h_ctrl = self.mid_block(h_base,h_ctrl, temb, cemb, attention_mask, cross_attention_kwargs) # todo umer: define self.mid_block

        # 3 - up
        for up in self.up_blocks: # todo umer: define self.up_blocks
            n_resnets = len(up.resnets)
            skips_hb = hs_base[-n_resnets:]
            skips_hc = hs_ctrl[-n_resnets:]
            hs_base = hs_base[:-n_resnets]
            hs_ctrl = hs_ctrl[:-n_resnets]
            h_base = up(h_base,h_ctrl,skips_hb,skips_hc,temb, cemb, attention_mask, cross_attention_kwargs)

        # 4 - conv out
        h_base = self.base_conv_norm_out(h_base)
        h_base = self.base_conv_act(h_base)
        h_base = self.base_conv_out(h_base)

        if not return_dict:
            return (h_base,)

        return ControlNetXSOutput(sample=h_base)


class ControlNetXSCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        base_in_channels: int,
        base_out_channels: int,
        ctrl_in_channels: int,
        ctrl_out_channels: int,
        temb_channels: int,
        max_norm_num_groups: Optional[int] = 32,
        has_crossattn=True,
        transformer_layers_per_block: Optional[Union[int, Tuple[int], Tuple[Tuple[int]]]] = 1,
        num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        add_downsample: bool = True,
        upcast_attention: Optional[bool] = False,
    ):
        super().__init__()
        base_resnets = []
        base_attentions = []
        ctrl_resnets =[]
        ctrl_attentions = []
        ctrl_to_base = []
        base_to_ctrl = []

        num_layers = 2 # only support sd + sdxl

        self.has_cross_attention = has_crossattn
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            base_in_channels = base_in_channels if i == 0 else base_out_channels
            ctrl_in_channels = ctrl_in_channels if i == 0 else ctrl_in_channels

            # Before the resnet/attention application, information is concatted from base to control.
            # Concat doesn't require change in number of channels
            base_to_ctrl.append(make_zero_conv(base_in_channels, base_in_channels))

            base_resnets.append(
                ResnetBlock2D(
                    in_channels=base_in_channels,
                    out_channels=base_out_channels,
                    temb_channels=temb_channels,
                )
            )
            ctrl_resnets.append(
                ResnetBlock2D(
                    in_channels=ctrl_in_channels,
                    out_channels=ctrl_in_channels,
                    temb_channels=temb_channels,
                    groups=find_largest_factor(ctrl_in_channels, max_factor=max_norm_num_groups),
                    groups_out=find_largest_factor(ctrl_in_channels, max_factor=max_norm_num_groups),
                    eps=1e-5,
                )
            )

            if has_crossattn:
                base_attentions.append(
                    Transformer2DModel(
                            num_attention_heads,
                            base_out_channels // num_attention_heads,
                            in_channels=base_out_channels,
                            num_layers=transformer_layers_per_block[i],
                            cross_attention_dim=cross_attention_dim,
                            use_linear_projection=True,
                            upcast_attention=upcast_attention,
                    )
                )
                ctrl_attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        ctrl_out_channels // num_attention_heads,
                        in_channels=ctrl_out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=True,
                        upcast_attention=upcast_attention,
                        norm_num_groups=find_largest_factor(ctrl_out_channels, max_factor=max_norm_num_groups),
                    )
                )

            # After the resnet/attention application, information is added from control to base
            # Addition requires change in number of channels
            ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))

        if add_downsample:
            # Before the downsampler application, information is concatted from base to control
            # Concat doesn't require change in number of channels
            base_to_ctrl.append(make_zero_conv(base_out_channels, base_out_channels))

            self.base_downsamplers = Downsample2D(base_out_channels, use_conv=True, out_channels=base_out_channels, name="op")
            self.ctrl_downsamplers = Downsample2D(ctrl_out_channels, use_conv=True, out_channels=ctrl_out_channels, name="op")

            # After the downsampler application, information is added from control to base
            # Addition requires change in number of channels
            ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
        else:
            self.base_downsamplers = None
            self.ctrl_downsamplers = None

        self.base_resnets = nn.ModuleList(base_resnets)
        self.ctrl_resnets = nn.ModuleList(ctrl_resnets)
        self.base_attentions = nn.ModuleList(base_attentions) if has_crossattn else [None]*num_layers
        self.ctrl_attentions = nn.ModuleList(ctrl_attentions) if has_crossattn else [None]*num_layers
        self.base_to_ctrl = nn.ModuleList(base_to_ctrl)
        self.ctrl_to_base = nn.ModuleList(ctrl_to_base)

        self.gradient_checkpointing = False

    @classmethod
    def from_modules(
        cls,
        base_resnets: List[ResnetBlock2D], ctrl_resnets: List[ResnetBlock2D],
        base_to_control_connections: List[nn.Conv2d], control_to_base_connections: List[nn.Conv2d],
        base_attentions: Optional[List[Transformer2DModel]] = None, ctrl_attentions: Optional[List[Transformer2DModel]] = None,
        base_downsampler: Optional[List[Transformer2DModel]] = None, ctrl_downsampler: Optional[List[Transformer2DModel]] = None,):
        """todo umer"""
        block = cls(
            in_channels = None,
            out_channels = None,
            temb_channels = None,
            max_norm_num_groups = 32,
            has_crossattn = True,
            transformer_layers_per_block = 1,
            num_attention_heads = 1,
            cross_attention_dim = 1024,
            add_downsample = True,
            upcast_attention = False,
        )

        block.base_resnets = base_resnets
        block.base_attentions = base_attentions
        block.ctrl_resnets = ctrl_resnets
        block.ctrl_attentions = ctrl_attentions
        block.b2c = base_to_control_connections
        block.c2b = control_to_base_connections
        block.base_downsampler = base_downsampler
        block.ctrl_downsampler = ctrl_downsampler

        return block

    def forward(
        self,
        hidden_states_base: torch.FloatTensor,
        hidden_states_ctrl: torch.FloatTensor,
        conditioning_scale: Optional[float] = 1.0,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]: # todo umer: output type hint correct?
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        h_base = hidden_states_base
        h_ctrl = hidden_states_ctrl

        base_output_states = ()
        ctrl_output_states = ()

        base_blocks = list(zip(self.base_resnets, self.base_attentions))
        ctrl_blocks = list(zip(self.ctrl_resnets, self.ctrl_attentions))

        for (b_res, b_attn), (c_res, c_attn), b2c, c2b in zip(base_blocks, ctrl_blocks, self.base_to_ctrl, self.ctrl_to_base):
            if self.training and self.gradient_checkpointing:
                raise NotImplementedError("todo umer")
            else:
                # concat base -> ctrl
                h_ctrl = torch.cat([h_ctrl, b2c(h_base)], dim=1)

                # apply base subblock
                h_base = b_res(h_base, temb)
                if b_attn is not None:
                    h_base = b_attn(
                        h_base,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                # apply ctrl subblock
                h_ctrl = c_res(h_ctrl, temb)
                if c_attn is not None:
                    h_ctrl = c_attn(
                        h_ctrl,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                # add ctrl -> base
                h_base = h_base + c2b(h_ctrl) * conditioning_scale

            base_output_states = base_output_states + (h_base,)
            ctrl_output_states = ctrl_output_states + (h_ctrl,)

        if self.base_downsamplers is not None:  # if we have a base_downsampler, then also a ctrl_downsampler
            b2c = self.base_to_ctrl[-1]
            c2b = self.ctrl_to_base[-1]

            # concat base -> ctrl
            h_ctrl = torch.cat([h_ctrl, b2c(h_base)], dim=1)
            # apply base subblock
            h_base = self.base_downsamplers(h_base)
            # apply ctrl subblock
            h_ctrl = self.ctrl_downsamplers(h_ctrl)
            # add ctrl -> base
            h_base = h_base + c2b(h_ctrl) * conditioning_scale

            base_output_states = base_output_states + (h_base,)
            ctrl_output_states = ctrl_output_states + (h_ctrl,)

        return h_base, h_ctrl,base_output_states, ctrl_output_states


class ControlNetXSCrossAttnMidBlock2D(nn.Module):
    def __init__(
        self,
        base_channels: int,
        ctrl_channels: int,
        temb_channels: Optional[int] = None,
        transformer_layers_per_block: int = 1,
        num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        upcast_attention: bool = False,
    ):
        super().__init__()

        # Before the midblock application, information is concatted from base to control.
        # Concat doesn't require change in number of channels
        self.base_to_ctrl = make_zero_conv(base_channels, base_channels)

        self.base_midblock = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=base_channels,
            temb_channels=temb_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            use_linear_projection=True,
            upcast_attention=upcast_attention
        )
        self.ctrl_midblock = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=ctrl_channels + base_channels,
            out_channels=ctrl_channels,
            temb_channels=temb_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads, # todo umer: n_attn_heads different for base / ctrl?
            use_linear_projection=True,
            upcast_attention=upcast_attention
        )

        # After the midblock application, information is added from control to base
        # Addition requires change in number of channels
        self.ctrl_to_base = make_zero_conv(ctrl_channels, base_channels)

        self.gradient_checkpointing = False

    @classmethod
    def from_modules(
        cls,
        resnet: ResnetBlock2D,
        attention: Optional[Transformer2DModel] = None,
        upsampler: Optional[Upsample2D] = None,
    ):
        """Create empty subblock and set resnet, attention and upsampler manually"""
        # todo umer
        subblock = cls()
        subblock.resnet = resnet
        subblock.attention = attention
        subblock.upsampler = upsampler
        subblock.in_channels = resnet.in_channels
        subblock.out_channels = resnet.out_channels
        return subblock

    def forward(
        self,
        hidden_states_base: torch.FloatTensor,
        hidden_states_ctrl: torch.FloatTensor,
        conditioning_scale: Optional[float] = 1.0,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor: # todo umer: output type hint correct?
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        h_base = hidden_states_base
        h_ctrl = hidden_states_ctrl

        joint_args = {
            "temb": temb,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
            "cross_attention_kwargs": cross_attention_kwargs,
            "encoder_attention_mask": encoder_attention_mask,
        }

        h_ctrl = torch.cat([h_ctrl, self.base_to_ctrl(h_base)], dim=1)  # concat base -> ctrl
        h_base = self.base_midblock(h_base, **joint_args)  # apply base mid block
        h_ctrl = self.ctrl_midblock(h_ctrl, **joint_args)  # apply ctrl mid block
        h_base = h_base + self.ctrl_to_base(h_ctrl) * conditioning_scale  # add ctrl -> base

        return h_base, h_ctrl


class ControlNetXSCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        ctrl_skip_channels: List[int],
        temb_channels: int,
        has_crossattn=True,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1024,
        add_upsample: bool = True,
        upcast_attention: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        ctrl_to_base = []

        num_layers = 3 # only support sd + sdxl

        self.has_cross_attention = has_crossattn
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )

            if has_crossattn:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=True,
                        upcast_attention=upcast_attention,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions) if has_crossattn else [None]*num_layers
        self.ctrl_to_base = nn.ModuleList(ctrl_to_base)

        if add_upsample:
            self.upsamplers = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    @classmethod
    def from_modules(
        cls,
        resnet: ResnetBlock2D,
        attention: Optional[Transformer2DModel] = None,
        upsampler: Optional[Upsample2D] = None,
    ):
        """Create empty subblock and set resnet, attention and upsampler manually"""
        # todo umer
        subblock = cls()
        subblock.resnet = resnet
        subblock.attention = attention
        subblock.upsampler = upsampler
        subblock.in_channels = resnet.in_channels
        subblock.out_channels = resnet.out_channels
        return subblock

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple_base: Tuple[torch.FloatTensor, ...],
        res_hidden_states_tuple_cltr: Tuple[torch.FloatTensor, ...],
        conditioning_scale: Optional[float] = 1.0,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor: # todo umer: output type hint correct?
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # In ControlNet-XS, the last resnet/attention and the upsampler are treated as a group.
        # So we separate them to pass information from ctrl to base correctly.
        if self.upsamplers is None:
            resnets_without_upsampler = self.resnets
            attn_without_upsampler = self.attentions
        else:
            resnets_without_upsampler = self.resnets[:-1]
            attn_without_upsampler = self.attentions[:-1]
            resnet_with_upsampler = self.resnets[-1]
            attn_with_upsampler = self.attentions[-1]

        for resnet, attn, c2b, res_h_base, res_h_ctrl in zip(resnets_without_upsampler, attn_without_upsampler, self.ctrl_to_base, reversed(res_hidden_states_tuple_base), reversed(res_hidden_states_tuple_cltr)):
            hidden_states += c2b(res_h_ctrl) * conditioning_scale
            hidden_states = torch.cat([hidden_states, res_h_base], dim=1)

            if self.training and self.gradient_checkpointing:
                raise NotImplementedError("todo umer")
            else:
                hidden_states = resnet(hidden_states, temb)
                if attn is not None:
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

        if self.upsampler is not None:
            c2b = self.ctrl_to_base[-1]
            res_h_base = res_hidden_states_tuple_base[0]
            res_h_ctrl = res_hidden_states_tuple_cltr[0]

            hidden_states += c2b(res_h_ctrl) * conditioning_scale
            hidden_states = torch.cat([hidden_states, res_h_base], dim=1)

            hidden_states = resnet_with_upsampler(hidden_states, temb)
            if attn_with_upsampler is not None:
                hidden_states = attn_with_upsampler(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            hidden_states = self.upsampler(hidden_states, upsample_size)

        return hidden_states


def make_zero_conv(in_channels, out_channels=None):
    return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def find_largest_factor(number, max_factor):
    factor = max_factor
    if factor >= number:
        return number
    while factor != 0:
        residual = number % factor
        if residual == 0:
            return factor
        factor -= 1
