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
from itertools import chain, zip_longest

import torch
import torch.nn as nn
import torch.utils.checkpoint

from torch.nn.modules.normalization import GroupNorm

from ..configuration_utils import ConfigMixin
from ..loaders import UNet2DConditionLoadersMixin
from ..utils import BaseOutput, logging

from .embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
    get_timestep_embedding
)
from .lora import LoRACompatibleConv
from .modeling_utils import ModelMixin
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    CrossAttnUpBlock2D,
    UpBlock2D,
    ResnetBlock2D,
    Transformer2DModel,
    Downsample2D,
    Upsample2D
)
from .unet_2d_condition import UNet2DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None


# Q: better name?
class ControlledUNet2DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),#note umer: not used everywhere by me. fix later.
            act_fn: str = "silu",
            time_embedding_type: str = "positional",
            time_embedding_dim: Optional[int] = None,
            time_embedding_act_fn: Optional[str] = None,
            timestep_post_act: Optional[str] = None,
            time_cond_proj_dim: Optional[int] = None,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0, 
            encoder_hid_dim: Optional[int] = 768, # Note Umer: should not be hard coded, but okay for minimal functional run - this comes from the text encoder output shape
            cross_attention_dim: Union[int, Tuple[int]] = 1280, # Note Umer: should not be hard coded, but okay for minimal functional run - this from the unet shapes
        ):
        super().__init__()

        # 1 - Save parameters
        # TODO make variables
        self.control_mode = "canny"
        self.learn_embedding = False
        self.infusion2control = "cat"
        self.infusion2base = "add"
        self.in_ch_factor = 1 if "cat" == 'add' else 2
        self.guiding = "encoder"
        self.two_stream_mode = "cross"
        self.control_model_ratio = 1.0
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.no_control = False
        self.control_scale = 1.0

        self.hint_model = None

        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift
        
        # Time embedding
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        # Text embedding
        self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)

        # 2 - Create base and control model
        self.base_model = base_model = UNet2DConditionModel(#todo make variable
            block_out_channels=(320, 640, 1280),
            down_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D","UpBlock2D"),
        )
        self.control_model = ctrl_model = UNet2DConditionModel(#todo make variable
            block_out_channels=[32,64,128],
            down_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D","UpBlock2D"),
            time_embedding_dim=1280
        ) # todo: make variable
        for i, extra_channels in enumerate(((320, 320), (320,640), (640,1280))): # todo: make variable (sth like zip(block_out_channels[:-1],block_out_channels[1:]))
            e1,e2=extra_channels
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=0, by=e1)
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=1, by=e2)
            if self.control_model.down_blocks[i].downsamplers: increase_block_input_in_encoder_downsampler(self.control_model, block_no=i, by=e2)
        increase_block_input_in_mid_resnet(self.control_model, by=1280) # todo: make var

        # 3 - Gather Channel Sizes
        ch_inout_ctrl = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        # 3.1 - input convolution
        ch_inout_ctrl['enc'].append((ctrl_model.conv_in.in_channels, ctrl_model.conv_in.out_channels))
        ch_inout_base['enc'].append((base_model.conv_in.in_channels, base_model.conv_in.out_channels))

        # 3.2 - encoder blocks
        for module in ctrl_model.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    ch_inout_ctrl['enc'].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    ch_inout_ctrl['enc'].append((module.downsamplers[0].channels, module.downsamplers[0].out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')
    
        for module in base_model.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    ch_inout_base['enc'].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    ch_inout_base['enc'].append((module.downsamplers[0].channels, module.downsamplers[0].out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')

        # 3.3 - middle block
        ch_inout_ctrl['mid'].append((ctrl_model.mid_block.resnets[0].in_channels, ctrl_model.mid_block.resnets[0].out_channels))
        ch_inout_base['mid'].append((base_model.mid_block.resnets[0].in_channels, base_model.mid_block.resnets[0].out_channels))
    
        # 3.4 - decoder blocks
        for module in base_model.up_blocks:
            if isinstance(module, (CrossAttnUpBlock2D, UpBlock2D)):
                for r in module.resnets:
                    ch_inout_base['dec'].append((r.in_channels, r.out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')
            
        self.ch_inout_ctrl = ch_inout_ctrl
        self.ch_inout_base = ch_inout_base

        # 4 - Build connections between base and control model
        self.enc_zero_convs_out = nn.ModuleList([])
        self.enc_zero_convs_in = nn.ModuleList([])

        self.middle_block_out = nn.ModuleList([])
        self.middle_block_in = nn.ModuleList([])

        self.dec_zero_convs_out = nn.ModuleList([])
        self.dec_zero_convs_in = nn.ModuleList([])

        for ch_io_base in ch_inout_base['enc']:
            self.enc_zero_convs_in.append(self.make_zero_conv(
                in_channels=ch_io_base[1], out_channels=ch_io_base[1])
            )
        
        self.middle_block_out = self.make_zero_conv(ch_inout_ctrl['mid'][-1][1], ch_inout_base['mid'][-1][1])
        
        self.dec_zero_convs_out.append(
            self.make_zero_conv(ch_inout_ctrl['enc'][-1][1], ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(ch_inout_ctrl['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(ch_inout_ctrl['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
            )
    
        # 5 - Input hint block TODO: Understand
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
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
            zero_module(nn.Conv2d(256, int(model_channels * self.control_model_ratio), 3, padding=1))
        )
    
        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list))


    def forward(self, x: torch.Tensor, t: torch.Tensor, encoder_hidden_states: torch.Tensor, c: dict, hint: torch.Tensor, no_control=False, **kwargs):
        """ Params from unet_2d_condition.UNet2DConditionModel.forward:
        # self,
        # sample: torch.FloatTensor,
        # timestep: Union[torch.Tensor, float, int],
        # encoder_hidden_states: torch.Tensor,
        # class_labels: Optional[torch.Tensor] = None,
        # timestep_cond: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        # down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        # mid_block_additional_residual: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        # return_dict: bool = True,
        """

        # # < from forward
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if x.size(0) // 2 == hint.size(0): hint = torch.cat([hint, hint], dim=0) # for classifier free guidance
        
        timesteps=t
        context=c.get("crossattn", None)
        y=c.get("vector", None)
        # # />

        # # < from forward_
        if no_control: return self.base_model(x=x, timesteps=timesteps, context=context, y=y, **kwargs)

        # time embeddings
        timesteps = timesteps[None]
        t_emb = get_timestep_embedding(
            timesteps, 
            self.model_channels,
            # # TODO: Undetrstand flip_sin_to_cos / (downscale_)freq_shift
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.freq_shift,
        )        
        # self.learn_embedding == False
        temb = self.base_model.time_embedding(t_emb)

        if y is not None: emb = emb + self.base_model.label_emb(y) # ?? - sth with class-conditioning
        # text embeddings
        cemb = self.encoder_hid_proj(encoder_hidden_states) # Q: use the base/ctrl models' encoder_hid_proj? Need to make sure dims fit

        emb = temb + cemb

        guided_hint = self.input_hint_block(hint)

        h_ctrl = h_base = x
        hs_base, hs_ctrl = [], []
        it_enc_convs_in, it_enc_convs_out, it_dec_convs_in, it_dec_convs_out = map(iter, (self.enc_zero_convs_in, self.enc_zero_convs_out, self.dec_zero_convs_in, self.dec_zero_convs_out))
        scales = iter(self.scale_list)

        base_down_subblocks = to_sub_blocks(self.base_model.down_blocks)
        ctrl_down_subblocks = to_sub_blocks(self.control_model.down_blocks)
        base_mid_subblocks = to_sub_blocks([self.base_model.mid_block])
        ctrl_mid_subblocks = to_sub_blocks([self.control_model.mid_block])
        base_up_subblocks = to_sub_blocks(self.base_model.up_blocks)

        # Cross Control
        # 0 - conv in
        h_base = self.base_model.conv_in(h_base)
        h_ctrl = self.control_model.conv_in(h_ctrl)
        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)
        # 1 - input blocks (encoder)
        for m_base, m_ctrl  in zip(base_down_subblocks, ctrl_down_subblocks):
            inp_base2ctrl = next(it_enc_convs_in)(h_base) # get info from base encoder 
            if guided_hint is not None: # in first, add hint info if it exists 
                inp_base2ctrl += guided_hint
                guided_hint = None
            h_ctrl = torch.cat([h_ctrl, inp_base2ctrl], dim=1)
            h_base = m_base(h_base, temb, cemb, context)
            h_ctrl = m_ctrl(h_ctrl, temb, cemb, context)
            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)
        # 2 - mid blocks (bottleneck)
        h_ctrl = torch.concat([h_ctrl, h_base], dim=1)
        for m_base, m_ctrl in zip(base_mid_subblocks, ctrl_mid_subblocks):
            h_base = m_base(h_base, temb, cemb, context)
            h_ctrl = m_ctrl(h_ctrl, temb, cemb, context)
        # 3 - output blocks (decoder)
        for m_base in base_up_subblocks:
            h_base = h_base + next(it_dec_convs_out)(hs_ctrl.pop()) * next(scales) # add info from ctrl encoder 
            h_base = torch.cat([h_base, hs_base.pop()], dim=1) # concat info from base encoder+ctrl encoder
            h_base = m_base(h_base, temb, cemb, context)
        return self.base_model.conv_out(h_base)


    def make_zero_conv(self, in_channels, out_channels=None):
        # keep running track # todo: better comment
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


def increase_block_input_in_encoder_resnet(unet, block_no, resnet_idx, by):
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


def increase_block_input_in_encoder_downsampler(unet, block_no, by):
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


def increase_block_input_in_mid_resnet(unet, by):
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


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


from diffusers.models.unet_2d_blocks import ResnetBlock2D, Transformer2DModel, Downsample2D, Upsample2D
class EmbedSequential(nn.ModuleList):
    """Sequential module passing embeddings (time and conditioning) to children if they support it."""
    def __init__(self,ms,*args,**kwargs):
        if not is_iterable(ms): ms = [ms]
        super().__init__(ms,*args,**kwargs)
    
    def forward(self,x,temb,cemb,context):
        for m in self:
            if isinstance(m,ResnetBlock2D): x=m(x,temb)
            elif isinstance(m,Transformer2DModel): x=m(x,cemb).sample # Q: Include temp also?
            elif isinstance(m,Downsample2D): x=m(x)
            elif isinstance(m,Upsample2D): x=m(x)
            else: raise ValueError(f'Type of m is {type(m)} but should be `ResnetBlock2D`, `Transformer2DModel`,  `Downsample2D`, `Upsample2D`')
        return x


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
