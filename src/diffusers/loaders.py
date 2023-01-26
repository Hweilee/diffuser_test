# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
from collections import defaultdict
from typing import Callable, Dict, Union, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .models.cross_attention import LoRACrossAttnProcessor
from .models.modeling_utils import _get_model_file
from .utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE, logging


logger = logging.get_logger(__name__)


LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"


class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = {k: v for k, v in enumerate(state_dict.keys())}
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = key.split(".processor")[0] + ".processor"
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class UNet2DConditionLoadersMixin:
    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into `UNet2DConditionModel`. Attention processor layers have to be
        defined in
        [cross_attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
        and be a `torch.nn.Module` class.

        <Tip warning={true}>

            This function is experimental and might change in the future

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>
        """

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", LORA_WEIGHT_NAME)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}

        is_lora = all("lora" in k for k in state_dict.keys())

        if is_lora:
            lora_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in lora_grouped_dict.items():
                rank = value_dict["to_k_lora.down.weight"].shape[0]
                cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
                hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

                attn_processors[key] = LoRACrossAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
                )
                attn_processors[key].load_state_dict(value_dict)

        else:
            raise ValueError(f"{model_file} does not seem to be in the correct format expected by LoRA training.")

        # set correct dtype & device
        attn_processors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in attn_processors.items()}

        # set layers
        self.set_attn_processor(attn_processors)

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weights_name: str = LORA_WEIGHT_NAME,
        save_function: Callable = None,
    ):
        r"""
        Save an attention procesor to a directory, so that it can be re-loaded using the
        `[`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]` method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = AttnProcsLayers(self.attn_processors)

        # Save the model
        state_dict = model_to_save.state_dict()

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "")
            if filename.startswith(weights_no_suffix) and os.path.isfile(full_filename) and is_main_process:
                os.remove(full_filename)

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weights_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")


class TextualInversionLoaderMixin:
    r"""
    Mixin class for adding textual inversion tokens and embeddings to the tokenizer and text encoder with method:
    - [`~TextualInversionLoaderMixin.load_textual_inversion_embeddings`]
    - [`~TextualInversionLoaderMixin.add_textual_inversion_embedding`]
    """

    def load_textual_inversion_embeddings(
        self, embedding_path_dict_or_list: Union[Dict[str, str], List[Dict[str, str]]], allow_replacement: bool = False
    ):
        r"""
        Loads textual inversion embeddings and adds them to the tokenizer's vocabulary and the text encoder's embeddings.

        Arguments:
            embeddings_path_dict_or_list (`Dict[str, str]` or `List[str]`):
                Dictionary of token to embedding path or List of embedding paths to embedding dictionaries.
                The dictionary must have the following keys:
                    - `token`: name of the token to be added to the tokenizers' vocabulary
                    - `embedding`: path to the embedding of the token to be added to the text encoder's embedding matrix
                The list must contain paths to embedding dictionaries where the keys are the tokens and the
                values are the embeddings (same as above dictionary definition).

        Returns:
            None
        """
        # Validate that inheriting class instance contains required attributes
        self._validate_method_call(self.load_textual_inversion_embeddings)

        if isinstance(embedding_path_dict_or_list, dict):
            for token, embedding_path in embedding_path_dict_or_list.items():
                # check if token in tokenizer vocab
                if token in self.tokenizer.get_vocab():
                    if allow_replacement:
                        logger.info(
                            f"Token {token} already in tokenizer vocabulary. Overwriting existing token and embedding with the new one."
                        )
                    else:
                        raise ValueError(
                            f"Token {token} already in tokenizer vocabulary. Please choose a different token name."
                        )

                embedding_dict = torch.load(embedding_path, map_location=self.text_encoder.device)
                embedding = self._extract_embedding_from_dict(embedding_dict)

                self.add_textual_inversion_embedding(token, embedding)

        elif isinstance(embedding_path_dict_or_list, list):
            for embedding_path in embedding_path_dict_or_list:
                embedding_dict = torch.load(embedding_path, map_location=self.text_encoder.device)
                token = self._extract_token_from_dict(embedding_dict)
                embedding = self._extract_embedding_from_dict(embedding_dict)

                # check if token in tokenizer vocab
                if token in self.tokenizer.get_vocab():
                    if allow_replacement:
                        logger.info(
                            f"Token {token} already in tokenizer vocabulary. Overwriting existing token and embedding with the new one."
                        )
                    else:
                        raise ValueError(
                            f"Token {token} already in tokenizer vocabulary. Please choose a different token name."
                        )
                self.add_textual_inversion_embedding(token, embedding)

    def add_textual_inversion_embedding(self, token: str, embedding: torch.Tensor):
        r"""
        Adds a token to the tokenizer's vocabulary and an embedding to the text encoder's embedding matrix.

        Arguments:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
            embedding (`torch.Tensor`):
                The embedding of the token to be added to the text encoder's embedding matrix
        """
        # NOTE: Not clear to me that we intend for this to be a public/exposed method.
        # Validate that inheriting class instance contains required attributes
        self._validate_method_call(self.load_textual_inversion_embeddings)

        embedding = embedding.to(self.text_encoder.dtype)

        if token in self.tokenizer.get_vocab():
            # If user has allowed replacement and the token exists, we only need to
            # extract the existing id and update the embedding
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
        else:
            # If the token does not exist, we add it to the tokenizer, then resize and update the
            # text encoder acccordingly
            self.tokenizer.add_tokens([token])

            token_id = self.tokenizer.convert_tokens_to_ids(token)
            # NOTE: len() does't start at 0, so we shouldn't need to +1
            # since we already updated the tokenizer and it's new length
            # should be old length + 1
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

    def _extract_embedding_from_dict(self, embedding_dict: Dict[str, str]) -> torch.Tensor:
        r"""
        Extracts the embedding from the embedding dictionary.

        Arguments:
            embedding_dict (`Dict[str, str]`):
                The embedding dictionary loaded from the embedding path

        Returns:
            embedding (`torch.Tensor`):
                The embedding to be added to the text encoder's embedding matrix
        """
        # auto1111 embedding case
        if "string_to_param" in embedding_dict:
            embedding_dict = embedding_dict["string_to_param"]
            embedding = embedding_dict["*"]
            return embedding

        return list(embedding_dict.values())[0]

    def _extract_token_from_dict(self, embedding_dict: Dict[str, str]) -> str:
        r"""
        Extracts the token from the embedding dictionary.

        Arguments:
            embedding_dict (`Dict[str, str]`):
                The embedding dictionary loaded from the embedding path

        Returns:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
        """
        # auto1111 embedding case
        if "string_to_param" in embedding_dict:
            token = embedding_dict["name"]
            return token

        return list(embedding_dict.keys())[0]

    def _validate_method_call(self, method: Callable):
        r"""
        Validates that the method is being called from a class instance that has the required attributes.

        Arguments:
            method (`function`):
                The class's method being called

        Raises:
            ValueError:
                If the method is being called from a class instance that does not have
                the required attributes, the method will not be callable.

        Returns:
            None
        """
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PreTrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling `{method.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder, PreTrainedModel):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `PreTrainedModel` for calling `{method.__name__}`"
            )
