# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Tuple

from ..models.attention_processor import (
    AttnProcessor2_0,
    PAGCFGIdentitySelfAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
)


class PAGMixin:
    r"""Mixin class for PAG."""

    def enable_pag(
        self,
        pag_scale: float = 0.0,
        pag_adaptive_scaling: float = 0.0,
        pag_applied_layers: Tuple[str] = ("mid",),  # ('down', 'mid', 'up',)
        pag_applied_layers_index: Tuple[str] = None,  # ('d4', 'd5', 'm0',)
        pag_cfg: bool = True,
    ):
        """Enables the FreeInit mechanism as in https://arxiv.org/abs/2312.07537.

        This implementation has been adapted from the [official repository](https://github.com/TianxingWu/FreeInit).

        Args:
            pag_scale (`float`, *optional*, defaults to `0.0`):
                Guidance scale of PAG.
        """
        self._pag_scale = pag_scale
        self._pag_adaptive_scaling = pag_adaptive_scaling
        self._pag_applied_layers = pag_applied_layers
        self._pag_applied_layers_index = pag_applied_layers_index
        self._pag_cfg = pag_cfg
        self._is_pag_enabled = True
        self._set_pag_attn_processor()

    def _get_self_attn_layers(self):
        down_layers = []
        mid_layers = []
        up_layers = []
        for name, module in self.unet.named_modules():
            if "attn1" in name and "to" not in name:
                layer_type = name.split(".")[0].split("_")[0]
                if layer_type == "down":
                    down_layers.append(module)
                elif layer_type == "mid":
                    mid_layers.append(module)
                elif layer_type == "up":
                    up_layers.append(module)
                else:
                    raise ValueError(f"Invalid layer type: {layer_type}")
        return up_layers, mid_layers, down_layers

    def _set_pag_attn_processor(self):
        up_layers, mid_layers, down_layers = self._get_self_attn_layers()

        if self._pag_cfg:
            replace_processor = PAGCFGIdentitySelfAttnProcessor2_0()
        else:
            replace_processor = PAGIdentitySelfAttnProcessor2_0()

        if self._pag_applied_layers_index:
            drop_layers = self._pag_applied_layers_index
            for drop_layer in drop_layers:
                layer_number = int(drop_layer[1:])
                try:
                    if drop_layer[0] == "d":
                        down_layers[layer_number].processor = replace_processor
                    elif drop_layer[0] == "m":
                        mid_layers[layer_number].processor = replace_processor
                    elif drop_layer[0] == "u":
                        up_layers[layer_number].processor = replace_processor
                    else:
                        raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                except IndexError:
                    raise ValueError(
                        f"Invalid layer index: {drop_layer}. Available layers: {len(down_layers)} down layers, {len(mid_layers)} mid layers, {len(up_layers)} up layers."
                    )
        elif self._pag_applied_layers:
            drop_full_layers = self._pag_applied_layers
            for drop_full_layer in drop_full_layers:
                try:
                    if drop_full_layer == "down":
                        for down_layer in down_layers:
                            down_layer.processor = replace_processor
                    elif drop_full_layer == "mid":
                        for mid_layer in mid_layers:
                            mid_layer.processor = replace_processor
                    elif drop_full_layer == "up":
                        for up_layer in up_layers:
                            up_layer.processor = replace_processor
                    else:
                        raise ValueError(f"Invalid layer type: {drop_full_layer}")
                except IndexError:
                    raise ValueError(
                        f"Invalid layer index: {drop_full_layer}. Available layers are: down, mid and up. If you need to specify each layer index, you can use `pag_applied_layers_index`"
                    )

    def _get_pag_scale(self, t):
        if self.do_pag_adaptive_scaling:
            signal_scale = self._pag_scale - self.pag_adaptive_scaling * (1000 - t)
            if signal_scale < 0:
                signal_scale = 0
            return signal_scale
        else:
            return self._pag_scale

    def _apply_perturbed_attention_guidance(self, noise_pred, do_classifier_free_guidance, guidance_scale, t):
        pag_scale = self._get_pag_scale(t)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_uncond)
                + pag_scale * (noise_pred_text - noise_pred_perturb)
            )
        else:
            noise_pred_uncond, noise_pred_perturb = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pag_scale * (noise_pred_uncond - noise_pred_perturb)
        return noise_pred

    def disable_pag(self):
        """Disables the PAG mechanism if enabled."""
        if not self.do_perturbed_attention_guidance:
            raise ValueError("PAG is not enabled.")

        up_layers, mid_layers, down_layers = self._get_self_attn_layers()
        if self._pag_applied_layers_index:
            drop_layers = self._pag_applied_layers_index
            for drop_layer in drop_layers:
                layer_number = int(drop_layer[1:])
                try:
                    if drop_layer[0] == "d":
                        down_layers[layer_number].processor = AttnProcessor2_0()
                    elif drop_layer[0] == "m":
                        mid_layers[layer_number].processor = AttnProcessor2_0()
                    elif drop_layer[0] == "u":
                        up_layers[layer_number].processor = AttnProcessor2_0()
                    else:
                        raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                except IndexError:
                    raise ValueError(
                        f"Invalid layer index: {drop_layer}. Available layers: {len(down_layers)} down layers, {len(mid_layers)} mid layers, {len(up_layers)} up layers."
                    )
        elif self._pag_applied_layers:
            drop_full_layers = self._pag_applied_layers
            for drop_full_layer in drop_full_layers:
                try:
                    if drop_full_layer == "down":
                        for down_layer in down_layers:
                            down_layer.processor = AttnProcessor2_0()
                    elif drop_full_layer == "mid":
                        for mid_layer in mid_layers:
                            mid_layer.processor = AttnProcessor2_0()
                    elif drop_full_layer == "up":
                        for up_layer in up_layers:
                            up_layer.processor = AttnProcessor2_0()
                    else:
                        raise ValueError(f"Invalid layer type: {drop_full_layer}")
                except IndexError:
                    raise ValueError(
                        f"Invalid layer index: {drop_full_layer}. Available layers are: down, mid and up. If you need to specify each layer index, you can use `pag_applied_layers_index`"
                    )
        self._pag_scale = None
        self._pag_adaptive_scaling = None
        self._pag_applied_layers = None
        self._pag_applied_layers_index = None
        self._pag_cfg = None
        self._is_pag_enabled = False

    @property
    def pag_adaptive_scaling(self):
        return self._pag_adaptive_scaling

    @property
    def do_pag_adaptive_scaling(self):
        return self._pag_adaptive_scaling > 0

    @property
    def do_perturbed_attention_guidance(self):
        return self._is_pag_enabled
