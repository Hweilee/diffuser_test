# Copyright 2024 MIT, Tsinghua University, NVIDIA CORPORATION and The HuggingFace Team.
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

from typing import Any, Optional, Callable, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from huggingface_hub import PyTorchModelHubMixin
import ipdb

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin

from ..activations import get_activation
from ..downsampling import ConvPixelUnshuffleDownsample2D, PixelUnshuffleChannelAveragingDownsample2D
from ..upsampling import ConvPixelShuffleUpsample2D, ChannelDuplicatingPixelUnshuffleUpsample2D, Upsample2D

from .vae import DecoderOutput


class RMSNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.empty(self.num_features, **factory_kwargs))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.empty(self.num_features, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super().__init__()

        padding = kernel_size // 2
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        if norm is None:
            self.norm = None
        elif norm == "rms2d":
            self.norm = RMSNorm2d(num_features=out_channels)
        elif norm == "bn2d":
            self.norm = BatchNorm2d(num_features=out_channels)
        else:
            raise ValueError(f"norm {norm} is not supported")
        self.act = get_activation(act_func) if act_func is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=(False, False, False),
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.glu_act = get_activation(act_func[1])
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=(False, False),
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.main = nn.Sequential(OrderedDict([
            ("conv1", ConvLayer(
                in_channels,
                mid_channels,
                kernel_size,
                stride,
                use_bias=use_bias[0],
                norm=norm[0],
                act_func=act_func[0],
            )),
            ("conv2", ConvLayer(
                mid_channels,
                out_channels,
                kernel_size,
                1,
                use_bias=use_bias[1],
                norm=norm[1],
                act_func=act_func[1],
            )),
        ]))
        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x) + self.shortcut(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=(False, False),
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        total_dim = heads * dim

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=scale // 2,
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = get_activation(kernel_func)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
    ):
        super().__init__()
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                    scales=scales,
                ),
                nn.Identity(),
            )
        else:
            raise ValueError(f"context_module {context_module} is not supported")
        if local_module == "GLUMBConv":
            self.local_module = ResidualBlock(
                GLUMBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                nn.Identity(),
            )
        else:
            raise NotImplementedError(f"local_module {local_module} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = get_activation(post_act) if post_act is not None else None

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
    elif block_type == "EViTGLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
    elif block_type == "EViTS5GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,))
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
        )
        stage.append(block)
    return stage


class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        latent_channels: int,
        width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        block_type: str | list[str] = "ResBlock",
        norm: str = "rms2d",
        act: str = "silu",
        downsample_block_type: str = "ConvPixelUnshuffle",
        downsample_match_channel: bool = True,
        downsample_shortcut: Optional[str] = "averaging",
        out_norm: Optional[str] = None,
        out_act: Optional[str] = None,
        out_shortcut: Optional[str] = "averaging",
        double_latent: bool = False,
    ):
        super().__init__()
        num_stages = len(width_list)
        self.num_stages = num_stages

        # validate config
        if len(depth_list) != num_stages or len(width_list) != num_stages:
            raise ValueError(f"len(depth_list) {len(depth_list)} and len(width_list) {len(width_list)} should be equal to num_stages {num_stages}")
        if not isinstance(block_type, (str, list)) or (isinstance(block_type, list) and len(block_type) != num_stages):
            raise ValueError(f"block_type should be either a str or a list of str with length {num_stages}, but got {block_type}")
        
        # project in
        if depth_list[0] > 0:
            project_in_block = nn.Conv2d(
                in_channels=in_channels,
                out_channels=width_list[0],
                kernel_size=3,
                padding=1,
            )
        elif depth_list[1] > 0:
            if downsample_block_type == "Conv":
                project_in_block = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=width_list[1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            elif downsample_block_type == "ConvPixelUnshuffle":
                project_in_block = ConvPixelUnshuffleDownsample2D(
                    in_channels=in_channels, out_channels=width_list[1], kernel_size=3, factor=2
                )
            else:
                raise ValueError(f"block_type {downsample_block_type} is not supported for downsampling")
        else:
            raise ValueError(f"depth list {depth_list} is not supported for encoder project in")
        self.project_in = project_in_block

        # stages
        self.stages: list[nn.Module] = []
        for stage_id, (width, depth) in enumerate(zip(width_list, depth_list)):
            stage_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage = build_stage_main(
                width=width, depth=depth, block_type=stage_block_type, norm=norm, act=act, input_width=width
            )
            if stage_id < num_stages - 1 and depth > 0:
                downsample_out_channels = width_list[stage_id + 1] if downsample_match_channel else width
                if downsample_block_type == "Conv":
                    downsample_block = nn.Conv2d(
                        in_channels=width,
                        out_channels=downsample_out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                elif downsample_block_type == "ConvPixelUnshuffle":
                    downsample_block = ConvPixelUnshuffleDownsample2D(
                        in_channels=width, out_channels=downsample_out_channels, kernel_size=3, factor=2
                    )
                else:
                    raise ValueError(f"downsample_block_type {downsample_block_type} is not supported for downsampling")
                if downsample_shortcut is None:
                    pass
                elif downsample_shortcut == "averaging":
                    shortcut_block = PixelUnshuffleChannelAveragingDownsample2D(
                        in_channels=width, out_channels=downsample_out_channels, factor=2
                    )
                    downsample_block = ResidualBlock(downsample_block, shortcut_block)
                else:
                    raise ValueError(f"shortcut {downsample_shortcut} is not supported for downsample")
                stage.append(downsample_block)
            self.stages.append(nn.Sequential(OrderedDict([("op_list", nn.Sequential(*stage))])))
        self.stages = nn.ModuleList(self.stages)

        # project out
        project_out_layers: list[nn.Module] = []
        if out_norm is None:
            pass
        elif out_norm == "rms2d":
            project_out_layers.append(RMSNorm2d(num_features=width_list[-1]))
        elif out_norm == "bn2d":
            project_out_layers.append(BatchNorm2d(num_features=width_list[-1]))
        else:
            raise ValueError(f"norm {out_norm} is not supported for encoder project out")
        if out_act is not None:
            project_out_layers.append(get_activation(out_act))
        project_out_out_channels = 2 * latent_channels if double_latent else latent_channels
        project_out_layers.append(ConvLayer(
            in_channels=width_list[-1],
            out_channels=project_out_out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
        ))
        project_out_block = nn.Sequential(OrderedDict([("op_list", nn.Sequential(*project_out_layers))]))
        if out_shortcut is None:
            pass
        elif out_shortcut == "averaging":
            shortcut_block = PixelUnshuffleChannelAveragingDownsample2D(
                in_channels=width_list[-1], out_channels=project_out_out_channels, factor=1
            )
            project_out_block = ResidualBlock(project_out_block, shortcut_block)
        else:
            raise ValueError(f"shortcut {out_shortcut} is not supported for encoder project out")
        self.project_out = project_out_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        latent_channels: int,
        in_shortcut: Optional[str] = "duplicating",
        width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        block_type: str | list[str] = "ResBlock",
        norm: str | list[str] = "rms2d",
        act: str | list[str] = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        upsample_match_channel: bool = True,
        upsample_shortcut: str = "duplicating",
        out_norm: str = "rms2d",
        out_act: str = "relu",
    ):
        super().__init__()
        num_stages = len(width_list)
        self.num_stages = num_stages

        # validate config
        if len(depth_list) != num_stages or len(width_list) != num_stages:
            raise ValueError(f"len(depth_list) {len(depth_list)} and len(width_list) {len(width_list)} should be equal to num_stages {num_stages}")
        if not isinstance(block_type, (str, list)) or (isinstance(block_type, list) and len(block_type) != num_stages):
            raise ValueError(f"block_type should be either a str or a list of str with length {num_stages}, but got {block_type}")
        if not isinstance(norm, (str, list)) or (isinstance(norm, list) and len(norm) != num_stages):
            raise ValueError(f"norm should be either a str or a list of str with length {num_stages}, but got {norm}")
        if not isinstance(act, (str, list)) or (isinstance(act, list) and len(act) != num_stages):
            raise ValueError(f"act should be either a str or a list of str with length {num_stages}, but got {act}")

        # project in
        project_in_block = ConvLayer(
            in_channels=latent_channels,
            out_channels=width_list[-1],
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
        )
        if in_shortcut is None:
            pass
        elif in_shortcut == "duplicating":
            shortcut_block = ChannelDuplicatingPixelUnshuffleUpsample2D(
                in_channels=latent_channels, out_channels=width_list[-1], factor=1
            )
            project_in_block = ResidualBlock(project_in_block, shortcut_block)
        else:
            raise ValueError(f"shortcut {in_shortcut} is not supported for decoder project in")
        self.project_in = project_in_block

        # stages
        self.stages: list[nn.Module] = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(width_list, depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_out_channels = width if upsample_match_channel else width_list[stage_id + 1]
                if upsample_block_type == "ConvPixelShuffle":
                    upsample_block = ConvPixelShuffleUpsample2D(
                        in_channels=width_list[stage_id + 1], out_channels=upsample_out_channels, kernel_size=3, factor=2
                    )
                elif upsample_block_type == "InterpolateConv":
                    upsample_block = Upsample2D(channels=width_list[stage_id + 1], use_conv=True, out_channels=upsample_out_channels)
                else:
                    raise ValueError(f"upsample_block_type {upsample_block_type} is not supported")
                if upsample_shortcut is None:
                    pass
                elif upsample_shortcut == "duplicating":
                    shortcut_block = ChannelDuplicatingPixelUnshuffleUpsample2D(
                        in_channels=width_list[stage_id + 1], out_channels=upsample_out_channels, factor=2
                    )
                    upsample_block = ResidualBlock(upsample_block, shortcut_block)
                else:
                    raise ValueError(f"shortcut {upsample_shortcut} is not supported for upsample")
                stage.append(upsample_block)

            stage_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage_norm = norm[stage_id] if isinstance(norm, list) else norm
            stage_act = act[stage_id] if isinstance(act, list) else act
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=stage_block_type,
                    norm=stage_norm,
                    act=stage_act,
                    input_width=(
                        width if upsample_match_channel else width_list[min(stage_id + 1, num_stages - 1)]
                    ),
                )
            )
            self.stages.insert(0, nn.Sequential(OrderedDict([("op_list", nn.Sequential(*stage))])))
        self.stages = nn.ModuleList(self.stages)

        # project out
        project_out_layers: list[nn.Module] = []
        if depth_list[0] > 0:
            project_out_in_channels = width_list[0]
        elif depth_list[1] > 0:
            project_out_in_channels = width_list[1]
        else:
            raise ValueError(f"depth list {depth_list} is not supported for decoder project out")
        if out_norm is None:
            pass
        elif out_norm == "rms2d":
            project_out_layers.append(RMSNorm2d(num_features=project_out_in_channels))
        elif out_norm == "bn2d":
            project_out_layers.append(BatchNorm2d(num_features=project_out_in_channels))
        else:
            raise ValueError(f"norm {out_norm} is not supported for decoder project out")
        project_out_layers.append(get_activation(out_act))
        if depth_list[0] > 0:
            project_out_layers.append(
                ConvLayer(
                    in_channels=project_out_in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    use_bias=True,
                    norm=None,
                    act_func=None,
                )
            )
        elif depth_list[1] > 0:
            if upsample_block_type == "ConvPixelShuffle":
                project_out_conv = ConvPixelShuffleUpsample2D(
                    in_channels=project_out_in_channels, out_channels=in_channels, kernel_size=3, factor=2
                )
            elif upsample_block_type == "InterpolateConv":
                project_out_conv = Upsample2D(channels=project_out_in_channels, use_conv=True, out_channels=in_channels)
            else:
                raise ValueError(f"upsample_block_type {upsample_block_type} is not supported for upsampling")

            project_out_layers.append(project_out_conv)
        else:
            raise ValueError(f"depth list {depth_list} is not supported for decoder project out")
        self.project_out = nn.Sequential(OrderedDict([("op_list", nn.Sequential(*project_out_layers))]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class DCAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        encoder_block_type: str | list[str] = "ResBlock",
        encoder_width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        encoder_depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        encoder_norm: str = "rms2d",
        encoder_act: str = "silu",
        downsample_block_type: str = "ConvPixelUnshuffle",
        decoder_block_type: str | list[str] = "ResBlock",
        decoder_width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        decoder_depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        decoder_norm: str = "rms2d",
        decoder_act: str = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        scaling_factor: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            width_list=encoder_width_list,
            depth_list=encoder_depth_list,
            block_type=encoder_block_type,
            norm=encoder_norm,
            act=encoder_act,
            downsample_block_type=downsample_block_type,
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            width_list=decoder_width_list,
            depth_list=decoder_depth_list,
            block_type=decoder_block_type,
            norm=decoder_norm,
            act=decoder_act,
            upsample_block_type=upsample_block_type,
        )

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (self.decoder.num_stages - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def decode(self, x: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        x = self.decoder(x)
        if not return_dict:
            return x
        else:
            return DecoderOutput(sample=x)

    def forward(self, x: torch.Tensor, global_step: int) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x, torch.tensor(0), {}
