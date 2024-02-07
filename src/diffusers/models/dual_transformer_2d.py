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
from ..utils import deprecate
from .transformers.dual_transformer_2d import DualTransformer2DModel


class DualTransformer2DModel(DualTransformer2DModel):
    deprecation_message = "Importing `DualTransformer2DModel` from `diffusers.models.dual_transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel`, instead."
    deprecate("DualTransformer2DModel", "0.29", deprecation_message)
