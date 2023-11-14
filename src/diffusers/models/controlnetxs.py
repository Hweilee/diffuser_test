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
from typing import Any, Dict, List, Optional, Union, Tuple

from itertools import zip_longest

import math
import torch
from torch import nn
from torch.nn.modules.normalization import GroupNorm
import torch.utils.checkpoint

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import UNet2DConditionLoadersMixin
from ..utils import BaseOutput, logging
from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from .embeddings import Timesteps
from .modeling_utils import ModelMixin
from .lora import LoRACompatibleConv
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    CrossAttnUpBlock2D,
    UpBlock2D,
    ResnetBlock2D,
    Transformer2DModel,
    Downsample2D,
    Upsample2D,
)
from .unet_2d_condition import UNet2DConditionModel
from ..umer_debug_logger import udl


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class ControlNetXSOutput(BaseOutput):
    """
    The output of [`ControlNetXSModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The output of the `ControlNetXSModel`. Unlike `ControlNetOutput` this is NOT to be added to
            the base model output, but is already the final output.
    """
    sample: torch.FloatTensor = None


# todo umer: add sth like FromOriginalControlnetMixin
class ControlNetXSModel(ModelMixin, ConfigMixin):
    r"""
    A ControlNet-XS model

    This model inherits from [`ModelMixin`] and [`ConfigMixin`].
    Check the superclass documentation for it's generic methods implemented for all models
    (such as downloading or saving).

    Most of parameters for this model are passed into the [`UNet2DConditionModel`] it creates.
    Check the documentation of [`UNet2DConditionModel`] for them.

    Parameters:
        conditioning_channels (`int`, defaults to 3):
            Number of channels of conditioning input (e.g. an image)
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        time_embedding_input_dim (`int`, defaults to 320):
            Dimension of input into time embedding. Needs to be same as in the base model.
        time_embedding_dim (`int`, defaults to 1280):
            Dimension of output from time embedding. Needs to be same as in the base model.
        learn_embedding (`bool`, defaults to `False`):
            Wether to use time embedding of the control model. If yes, the time embedding
            is a linear interpolation of the time embeddings of the control and base model
            with interpolation parameter `time_embedding_mix**3`.
        time_embedding_mix (`float`, defaults to 1.0):
            Linear interpolation parameter used if `learn_embedding` is `True`.
        base_model_channel_sizes (`Dict[str, List[Tuple[int]]]`):
            Channel sizes of each subblock of base model. Use `gather_subblock_sizes` on
            your base model to compute it.
    """

    # to delete later
    @classmethod
    def create_as_in_paper(cls, base_model: UNet2DConditionModel):
        return ControlNetXSModel.from_unet(
            base_model,
            time_embedding_mix =  0.95,
            learn_embedding = True,
            control_model_size_ratio = 0.1,
            dim_attention_heads = 64
        )

    @classmethod
    def gather_subblock_sizes(cls, unet: UNet2DConditionModel, base_or_control):
        if base_or_control not in ['base', 'control']:
            raise ValueError(f"`base_or_control` needs to be either `base` or `control`")

        channel_sizes = {'down': [], 'mid': [], 'up': []}

        # input convolution
        channel_sizes['down'].append((unet.conv_in.in_channels, unet.conv_in.out_channels))

        # encoder blocks
        for module in unet.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    channel_sizes['down'].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    channel_sizes['down'].append((module.downsamplers[0].channels, module.downsamplers[0].out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')

        # middle block
        channel_sizes['mid'].append((unet.mid_block.resnets[0].in_channels, unet.mid_block.resnets[0].out_channels))

        # decoder blocks
        if base_or_control == 'base':
            for module in unet.up_blocks:
                if isinstance(module, (CrossAttnUpBlock2D, UpBlock2D)):
                    for r in module.resnets:
                        channel_sizes['up'].append((r.in_channels, r.out_channels))
                else:
                   raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')

        return channel_sizes

    @register_to_config
    def __init__(
        self,      
        conditioning_channels: int = 3,
        controlnet_conditioning_channel_order: str = "rgb",
        time_embedding_input_dim: int = 320,
        time_embedding_dim: int = 1280,
        time_embedding_mix:float=1.0,
        learn_embedding: bool =False,
        base_model_channel_sizes: Dict[str, List[Tuple[int]]]={
            'down':[(4, 320), (320, 320), (320, 320), (320, 320), (320, 640), (640, 640), (640, 640), (640, 1280), (1280, 1280)],
            'mid': [(1280, 1280)],
            'up':  [(2560, 1280), (2560, 1280), (1920, 1280), (1920, 640), (1280, 640), (960, 640), (960, 320), (640, 320), (640, 320)]
        },
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        attention_type: str = "default",
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
    ):
        super().__init__()

        # 1 - Create control unet
        self.control_model = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            attention_type=attention_type,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )       

        # 2 - Do model surgery on control model
        # 2.1 - Allow to use the same time information as the base model
        adjust_time_dims(self.control_model, time_embedding_input_dim, time_embedding_dim)
 
        # 2.2 - Allow for information infusion from base model
        # todo umer: the assumption that block sizes = changing subblock sizes is false, eg when we have consecutive blocks of same size
        base_block_out_channels = [sz[1] for sz in base_model_channel_sizes['down'] if sz[0] != sz[1]]

        extra_channels = list(zip(
            base_block_out_channels[0:1] + base_block_out_channels[:-1],
            base_block_out_channels
        ))
        for i, (e1, e2) in enumerate(extra_channels):
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=0, by=e1)
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=1, by=e2)
            if self.control_model.down_blocks[i].downsamplers: increase_block_input_in_encoder_downsampler(self.control_model, block_no=i, by=e2)
        increase_block_input_in_mid_resnet(self.control_model, by=base_block_out_channels[-1])

        # 3 - Gather Channel Sizes
        self.ch_inout_ctrl = ControlNetXSModel.gather_subblock_sizes(self.control_model, base_or_control='control')
        self.ch_inout_base = base_model_channel_sizes

        # 4 - Build connections between base and control model
        self.down_zero_convs_out = nn.ModuleList([])
        self.down_zero_convs_in = nn.ModuleList([])
        self.middle_block_out = nn.ModuleList([])
        self.middle_block_in = nn.ModuleList([])
        self.up_zero_convs_out = nn.ModuleList([])
        self.up_zero_convs_in = nn.ModuleList([])

        for ch_io_base in self.ch_inout_base['down']:
            self.down_zero_convs_in.append(self.make_zero_conv(
                in_channels=ch_io_base[1], out_channels=ch_io_base[1])
            )
        for i in range(len(self.ch_inout_ctrl['down'])):
            self.down_zero_convs_out.append(
                self.make_zero_conv(self.ch_inout_ctrl['down'][i][1], self.ch_inout_base['down'][i][1])
            )       
 
        self.middle_block_out = self.make_zero_conv(self.ch_inout_ctrl['mid'][-1][1], self.ch_inout_base['mid'][-1][1])
        
        self.up_zero_convs_out.append(
            self.make_zero_conv(self.ch_inout_ctrl['down'][-1][1], self.ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(self.ch_inout_ctrl['down'])):
            self.up_zero_convs_out.append(
                self.make_zero_conv(self.ch_inout_ctrl['down'][-(i + 1)][1], self.ch_inout_base['up'][i - 1][1])
            )

        # 5 - Create conditioning hint embedding
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, block_out_channels[0], 3, padding=1))
        )
    
        # In the mininal implementation setting, we only need the control model up to the mid block
        del self.control_model.up_blocks
        del self.control_model.conv_norm_out
        del self.control_model.conv_out

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        conditioning_channels: int = 3,
        controlnet_conditioning_channel_order: str = "rgb",
        learn_embedding: bool = False,
        time_embedding_mix: float =  1.0,
        block_out_channels: Optional[Tuple[int]] = None,
        control_model_size_ratio: Optional[float] = None,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dim_attention_heads: Optional[int] = None
    ):
        r"""
        Instantiate a [`ControlNetXSModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model we want to control. The dimensions of the ControlNetXSModel will be adapted to it.
            conditioning_channels (`int`, defaults to 3):
                Number of channels of conditioning input (e.g. an image)
            controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
                The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
            learn_embedding (`bool`, defaults to `False`):
                Wether to use time embedding of the control model. If yes, the time embedding
                is a linear interpolation of the time embeddings of the control and base model
                with interpolation parameter `time_embedding_mix**3`.
            time_embedding_mix (`float`, defaults to 1.0):
                Linear interpolation parameter used if `learn_embedding` is `True`.
            block_out_channels (`Tuple[int]`, *optional*):
                Down blocks output channels in control model. Either this or `block_out_channels` must be given.
            control_model_size_ratio (float, *optional*):
                When given, block_out_channels is set to a relative fraction of the base model's block_out_channels.
                Either this or `control_model_size_ratio` must be given.
        """

        # check input
        fixed_size = block_out_channels is not None
        relative_size = control_model_size_ratio is not None
        if not (fixed_size ^ relative_size):
            raise ValueError("Pass exactly one of `block_out_channels` (for absolute sizing) or `control_model_ratio` (for relative sizing).")

        if num_attention_heads is not None and dim_attention_heads is not None:
            raise ValueError("Pass only one of `num_attention_heads` or `dim_attention_heads`.")

        # create model
        if block_out_channels is None:
            block_out_channels = [int(control_model_size_ratio*c) for c in unet.config.block_out_channels]

        if dim_attention_heads is not None:
            num_attention_heads = [math.ceil(c/dim_attention_heads) for c in block_out_channels]

        def get_time_emb_input_dim(unet: UNet2DConditionModel):return unet.time_embedding.linear_1.in_features
        def get_time_emb_dim(unet: UNet2DConditionModel): return unet.time_embedding.linear_2.out_features

        kwargs = dict(unet.config)
        kwargs.update(block_out_channels=block_out_channels)
        if num_attention_heads is not None:
            kwargs.update(attention_head_dim=num_attention_heads)

        to_remove = (
            'flip_sin_to_cos','freq_shift',
            'addition_embed_type','addition_time_embed_dim',
            'class_embed_type', 'num_class_embeds', 'projection_class_embeddings_input_dim', 'class_embeddings_concat'
        )
        for o in to_remove:
            del kwargs[o]

        kwargs.update(
            conditioning_channels = conditioning_channels,
            controlnet_conditioning_channel_order = controlnet_conditioning_channel_order,
            time_embedding_input_dim = get_time_emb_input_dim(unet),
            time_embedding_dim = get_time_emb_dim(unet),
            time_embedding_mix =  time_embedding_mix,
            learn_embedding = learn_embedding,
            base_model_channel_sizes = ControlNetXSModel.gather_subblock_sizes(unet, base_or_control='base'),
        )

        return cls(**kwargs)

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
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
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
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
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    # Copied from diffusers.models.controlnet.ControlNetModel._set_gradient_checkpointing
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        base_model: UNet2DConditionModel,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        guess_mode: bool = False, # todo umer: understand and implement if required
        return_dict: bool = True,
    ) -> Union[ControlNetXSOutput, Tuple]:
        """
        The [`ControlNetModel`] forward method.

        Args:
            base_model (`UNet2DConditionModel`):
                The base unet model we want to control.
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
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            # guess_mode (`bool`, defaults to `False`):
                # todo umer
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnetxs.ControlNetXSOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnetxs.ControlNetXSOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # scale control strength
        n_connections = len(self.down_zero_convs_out) + 1 + len(self.up_zero_convs_out)
        scale_list = torch.full((n_connections,), conditioning_scale)
   
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps=timestep
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

        t_emb = base_model.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.learn_embedding:
            ctrl_temb = self.control_model.time_embedding(t_emb, timestep_cond)
            base_temb = base_model.time_embedding(t_emb, timestep_cond)
            interpolation_param = self.config.time_embedding_mix ** 0.3

            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = base_model.time_embedding(t_emb)

        # added time & text embeddings
        aug_emb = None

        if base_model.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if base_model.config.class_embed_type == "timestep":
                class_labels = base_model.time_proj(class_labels)

            class_emb = base_model.class_embedding(class_labels).to(dtype=self.dtype)
            temb = temb + class_emb

        if base_model.config.addition_embed_type is not None:
            if base_model.config.addition_embed_type == "text":
                aug_emb = base_model.add_embedding(encoder_hidden_states)
            elif base_model.config.addition_embed_type == "text_image":
                raise NotImplementedError()
            elif base_model.config.addition_embed_type == "text_time":
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
                time_embeds = base_model.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(temb.dtype)
                aug_emb = base_model.add_embedding(add_embeds)
            elif base_model.config.addition_embed_type == "image":
                raise NotImplementedError()
            elif base_model.config.addition_embed_type == "image_hint":
                raise NotImplementedError()

        temb = temb + aug_emb if aug_emb is not None else temb

        # text embeddings
        cemb = encoder_hidden_states

        # Preparation
        guided_hint = self.input_hint_block(controlnet_cond)

        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []
        it_down_convs_in, it_down_convs_out, it_dec_convs_in, it_up_convs_out = map(iter, (self.down_zero_convs_in, self.down_zero_convs_out, self.up_zero_convs_in, self.up_zero_convs_out))
        scales = iter(scale_list)

        base_down_subblocks = to_sub_blocks(base_model.down_blocks)
        ctrl_down_subblocks = to_sub_blocks(self.control_model.down_blocks)
        base_mid_subblocks = to_sub_blocks([base_model.mid_block])
        ctrl_mid_subblocks = to_sub_blocks([self.control_model.mid_block])
        base_up_subblocks = to_sub_blocks(base_model.up_blocks)

        # Cross Control
        # 0 - conv in
        h_base = base_model.conv_in(h_base)
        h_ctrl = self.control_model.conv_in(h_ctrl)
        if guided_hint is not None: h_ctrl += guided_hint
        h_base = h_base + next(it_down_convs_out)(h_ctrl) * next(scales)

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        # 1 - down
        for m_base, m_ctrl  in zip(base_down_subblocks, ctrl_down_subblocks):
            h_ctrl = torch.cat([h_ctrl, next(it_down_convs_in)(h_base)], dim=1)  # A - concat base -> ctrl
            h_base = m_base(                                                     # B - apply base subblock
                h_base, temb, cemb,
                attention_mask, cross_attention_kwargs
            )                                 
            h_ctrl = m_ctrl(                                                     # C - apply ctrl subblock
                h_ctrl, temb, cemb,
                attention_mask, cross_attention_kwargs
            )                             
            h_base = h_base + next(it_down_convs_out)(h_ctrl) * next(scales)     # D - add ctrl -> base

            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)

        # 2 - mid
        h_ctrl = torch.cat([h_ctrl, next(it_down_convs_in)(h_base)], dim=1)      # A - concat base -> ctrl
        for m_base, m_ctrl in zip(base_mid_subblocks, ctrl_mid_subblocks):
            h_base = m_base(                                                     # B - apply base subblock
                h_base, temb, cemb,
                attention_mask, cross_attention_kwargs
            )  
            h_ctrl  = m_ctrl(                                                    # C - apply ctrl subblock
                h_ctrl, temb, cemb,
                attention_mask, cross_attention_kwargs
            )  
        h_base = h_base + self.middle_block_out(h_ctrl) * next(scales)           # D - add ctrl -> base
 
        # 3 - up
        for m_base in base_up_subblocks:
            h_base = h_base + next(it_up_convs_out)(hs_ctrl.pop()) * next(scales) # add info from ctrl encoder 
            h_base = torch.cat([h_base, hs_base.pop()], dim=1)                    # concat info from base encoder+ctrl encoder
            h_base = m_base(                                                   
                h_base, temb, cemb,
                attention_mask, cross_attention_kwargs
            )  

        h_base = base_model.conv_norm_out(h_base)
        h_base = base_model.conv_act(h_base)
        h_base = base_model.conv_out(h_base)

        if not return_dict:
            return h_base
        
        return ControlNetXSOutput(sample=h_base)

    def make_zero_conv(self, in_channels, out_channels=None):
        # keep running track of channels sizes
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


class EmbedSequential(nn.ModuleList):
    """Sequential module passing embeddings (time and conditioning) to children if they support it."""
    def __init__(self,ms,*args,**kwargs):
        if not is_iterable(ms): ms = [ms]
        super().__init__(ms,*args,**kwargs)
    
    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cemb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for m in self:
            if isinstance(m,ResnetBlock2D):
                x = m(x,temb)
            elif isinstance(m,Transformer2DModel):
                x = m(x,cemb,attention_mask=attention_mask,cross_attention_kwargs=cross_attention_kwargs).sample
            elif isinstance(m,Downsample2D):
                x = m(x)
            elif isinstance(m,Upsample2D):
                x = m(x)
            else:
                raise ValueError(f'Type of m is {type(m)} but should be `ResnetBlock2D`, `Transformer2DModel`,  `Downsample2D` or `Upsample2D`')

        return x


def adjust_time_dims(unet: UNet2DConditionModel, in_dim: int, out_dim: int):
    unet.time_embedding.linear_1 = nn.Linear(in_dim, out_dim)


def increase_block_input_in_encoder_resnet(unet:UNet2DConditionModel, block_no, resnet_idx, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    r=unet.down_blocks[block_no].resnets[resnet_idx]
    old_norm1, old_conv1, old_conv_shortcut = r.norm1,r.conv1,r.conv_shortcut
    # norm
    norm_args = 'num_groups num_channels eps affine'.split(' ')
    for a in norm_args: assert hasattr(old_norm1, a)
    norm_kwargs = { a: getattr(old_norm1, a) for a in norm_args }
    norm_kwargs['num_channels'] += by  # surgery done here
    # conv1
    conv1_args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in conv1_args: assert hasattr(old_conv1, a)
    conv1_kwargs = { a: getattr(old_conv1, a) for a in conv1_args }
    conv1_kwargs['bias'] = 'bias' in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs['in_channels'] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work) 
    conv_shortcut_args_kwargs = { 
        'in_channels': conv1_kwargs['in_channels'],
        'out_channels': conv1_kwargs['out_channels'],
        # default arguments from resnet.__init__
        'kernel_size':1, 
        'stride':1, 
        'padding':0,
        'bias':True
    }
    # swap old with new modules
    unet.down_blocks[block_no].resnets[resnet_idx].norm1 = GroupNorm(**norm_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].conv1 = LoRACompatibleConv(**conv1_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].conv_shortcut = LoRACompatibleConv(**conv_shortcut_args_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].in_channels += by  # surgery done here


def increase_block_input_in_encoder_downsampler(unet:UNet2DConditionModel, block_no, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    old_down=unet.down_blocks[block_no].downsamplers[0].conv
    # conv1
    args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in args: assert hasattr(old_down, a)
    kwargs = { a: getattr(old_down, a) for a in args}
    kwargs['bias'] = 'bias' in kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    kwargs['in_channels'] += by  # surgery done here
    # swap old with new modules
    unet.down_blocks[block_no].downsamplers[0].conv = LoRACompatibleConv(**kwargs)
    unet.down_blocks[block_no].downsamplers[0].channels += by  # surgery done here


def increase_block_input_in_mid_resnet(unet:UNet2DConditionModel, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    m=unet.mid_block.resnets[0]
    old_norm1, old_conv1, old_conv_shortcut = m.norm1,m.conv1,m.conv_shortcut
    # norm
    norm_args = 'num_groups num_channels eps affine'.split(' ')
    for a in norm_args: assert hasattr(old_norm1, a)
    norm_kwargs = { a: getattr(old_norm1, a) for a in norm_args }
    norm_kwargs['num_channels'] += by  # surgery done here
    # conv1
    conv1_args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in conv1_args: assert hasattr(old_conv1, a)
    conv1_kwargs = { a: getattr(old_conv1, a) for a in conv1_args }
    conv1_kwargs['bias'] = 'bias' in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs['in_channels'] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work) 
    conv_shortcut_args_kwargs = { 
        'in_channels': conv1_kwargs['in_channels'],
        'out_channels': conv1_kwargs['out_channels'],
        # default arguments from resnet.__init__
        'kernel_size':1, 
        'stride':1, 
        'padding':0,
        'bias':True
    }
    # swap old with new modules
    unet.mid_block.resnets[0].norm1 = GroupNorm(**norm_kwargs)
    unet.mid_block.resnets[0].conv1 = LoRACompatibleConv(**conv1_kwargs)
    unet.mid_block.resnets[0].conv_shortcut = LoRACompatibleConv(**conv_shortcut_args_kwargs)
    unet.mid_block.resnets[0].in_channels += by  # surgery done here


def is_iterable(o):
    if isinstance(o, str): return False
    try:
        iter(o)
        return True
    except TypeError:
        return False


def to_sub_blocks(blocks):
    if not is_iterable(blocks): blocks = [blocks]
    sub_blocks = []
    for b in blocks:
        current_subblocks = []
        if hasattr(b, 'resnets'):
            if hasattr(b, 'attentions') and b.attentions is not None:
                current_subblocks = list(zip_longest(b.resnets, b.attentions))
                 # if we have 1 more resnets than attentions, let the last subblock only be the resnet, not (resnet, None)
                if current_subblocks[-1][1] is None:
                    current_subblocks[-1] = current_subblocks[-1][0]
            else:
                current_subblocks = list(b.resnets)
        # upsamplers are part of the same block # q: what if we have multiple upsamplers?
        if hasattr(b, 'upsamplers') and b.upsamplers is not None: current_subblocks[-1] = list(current_subblocks[-1]) + list(b.upsamplers)
        # downsamplers are own block
        if hasattr(b, 'downsamplers') and b.downsamplers is not None: current_subblocks.append(list(b.downsamplers))   
        sub_blocks += current_subblocks
    return list(map(EmbedSequential, sub_blocks))


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
