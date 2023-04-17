from typing import Optional, Union

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.attention import AdaLayerNorm, FeedForward
from ...models.attention_processor import Attention
from ...models.embeddings import PatchEmbed, TimestepEmbedding, Timesteps
from ...models.transformer_2d import Transformer2DModelOutput


class SkipBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.skip_linear = nn.Linear(2 * dim, dim)

        # Use torch.nn.LayerNorm for now, following the original code
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, skip):
        x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = self.norm(x)

        return x


# Modified to support both pre-LayerNorm and post-LayerNorm configurations
# Don't support AdaLayerNormZero for now
# Modified from diffusers.models.attention.BasicTransformerBlock
class UTransformerBlock(nn.Module):
    r"""
    A modification of BasicTransformerBlock which supports pre-LayerNorm and post-LayerNorm configurations.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        pre_layer_norm: bool = True,
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        self.pre_layer_norm = pre_layer_norm

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # 1. Self-Attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        # elif self.use_ada_layer_norm_zero:
        #     self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            self.norm2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # Pre-LayerNorm
        if self.pre_layer_norm:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            else:
                norm_hidden_states = self.norm1(hidden_states)
        else:
            norm_hidden_states = hidden_states

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        # Post-LayerNorm
        if not self.pre_layer_norm:
            if self.use_ada_layer_norm:
                attn_output = self.norm1(attn_output, timestep)
            else:
                attn_output = self.norm1(hidden_states)

        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            # Pre-LayerNorm
            if self.pre_layer_norm:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
            else:
                norm_hidden_states = hidden_states
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            # Post-LayerNorm
            if not self.pre_layer_norm:
                attn_output = self.norm2(attn_output, timestep) if self.use_ada_layer_norm else self.norm2(attn_output)

            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        # Pre-LayerNorm
        if self.pre_layer_norm:
            norm_hidden_states = self.norm3(hidden_states)
        else:
            norm_hidden_states = hidden_states

        ff_output = self.ff(norm_hidden_states)

        # Post-LayerNorm
        if not self.pre_layer_norm:
            ff_output = self.norm3(ff_output)

        hidden_states = ff_output + hidden_states

        return hidden_states


# Modified from diffusers.models.transformer_2d.Transformer2DModel
# Modify the transformer block structure to be U-Net like following U-ViT
# Only supports patch-style input and torch.nn.LayerNorm currently
# https://github.com/baofff/U-ViT
class UTransformer2DModel(ModelMixin, ConfigMixin):
    """
    Transformer model based on the [U-ViT](https://github.com/baofff/U-ViT) architecture for image-like data. Compared
    to [`Transformer2DModel`], this model has skip connections between transformer blocks in a "U"-shaped fashion,
    similar to a U-Net. Takes either discrete (classes of vector embeddings) or continuous (actual embeddings) inputs.

    When input is continuous: First, project the input (aka embedding) and reshape to b, t, d. Then apply standard
    transformer action. Finally, reshape to image.

    When input is discrete: First, input (classes of latent pixels) is converted to embeddings and has positional
    embeddings applied, see `ImagePositionalEmbeddings`. Then apply standard transformer action. Finally, predict
    classes of unnoised image.

    Note that it is assumed one of the input classes is the masked latent pixel. The predicted classes of the unnoised
    image do not contain a prediction for the masked pixel as the unnoised image cannot be masked.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        pre_layer_norm: bool = False,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Input
        # Only support patch input of shape (batch_size, num_channels, height, width) for now
        assert in_channels is not None and patch_size is not None, "Patch input requires in_channels and patch_size."

        assert sample_size is not None, "UTransformer2DModel over patched input must provide sample_size"

        # 2. Define input layers
        self.height = sample_size
        self.width = sample_size

        self.patch_size = patch_size
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
        )

        # 3. Define transformers blocks
        # Modify this to have in_blocks ("downsample" blocks, even though we don't actually downsample), a mid_block,
        # and out_blocks ("upsample" blocks). Like a U-Net, there are skip connections from in_blocks to out_blocks in
        # a "U"-shaped fashion (e.g. first in_block to last out_block, etc.).
        self.transformer_in_blocks = nn.ModuleList(
            [
                UTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    pre_layer_norm=pre_layer_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers // 2)
            ]
        )

        self.transformer_mid_block = UTransformerBlock(
            inner_dim,
            num_attention_heads,
            attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            pre_layer_norm=pre_layer_norm,
            norm_elementwise_affine=norm_elementwise_affine,
        )

        # For each skip connection, we use a SkipBlock (concatenation + Linear + LayerNorm) to process the inputs
        # before each transformer out_block.
        self.transformer_out_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "skip": SkipBlock(
                            inner_dim,
                        ),
                        "block": UTransformerBlock(
                            inner_dim,
                            num_attention_heads,
                            attention_head_dim,
                            dropout=dropout,
                            cross_attention_dim=cross_attention_dim,
                            activation_fn=activation_fn,
                            num_embeds_ada_norm=num_embeds_ada_norm,
                            attention_bias=attention_bias,
                            only_cross_attention=only_cross_attention,
                            upcast_attention=upcast_attention,
                            norm_type=norm_type,
                            pre_layer_norm=pre_layer_norm,
                            norm_elementwise_affine=norm_elementwise_affine,
                        ),
                    }
                )
                for d in range(num_layers // 2)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
        hidden_states_is_embedding: bool = False,
        unpatchify: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            hidden_states_is_embedding (`bool`, *optional*, defaults to `False`):
                Whether or not hidden_states is an embedding directly usable by the transformer. In this case we will
                ignore input handling (e.g. continuous, vectorized, etc.) and directly feed hidden_states into the
                transformer blocks.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # 0. Check inputs

        if not unpatchify and return_dict:
            raise ValueError(
                f"Cannot both define `unpatchify`: {unpatchify} and `return_dict`: {return_dict} since when"
                f" `unpatchify` is {unpatchify} the returned output is of shape (batch_size, seq_len, hidden_dim)"
                " rather than (batch_size, num_channels, height, width)."
            )

        # 1. Input
        if not hidden_states_is_embedding:
            hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks

        # In ("downsample") blocks
        skips = []
        for in_block in self.transformer_in_blocks:
            hidden_states = in_block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
            skips.append(hidden_states)

        # Mid block
        hidden_states = self.transformer_mid_block(hidden_states)

        # Out ("upsample") blocks
        for out_block in self.transformer_out_blocks:
            hidden_states = out_block["skip"](hidden_states, skips.pop())
            hidden_states = out_block["block"](
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        # TODO: cleanup!
        # Don't support AdaLayerNorm for now, so no conditioning/scale/shift logic
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if unpatchify:
            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
        else:
            output = hidden_states

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class UniDiffuserModel(ModelMixin, ConfigMixin):
    """
    Transformer model for a image-text [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) model. This is a
    modification of [`UTransformer2DModel`] with input and output heads for the VAE-embedded latent image, the
    CLIP-embedded image, and the CLIP-embedded prompt (see paper for more details).

    Parameters:
        text_dim (`int`): The hidden dimension of the CLIP text model used to embed images.
        clip_img_dim (`int`): The hidden dimension of the CLIP vision model used to embed prompts.
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        text_dim: int = 768,
        clip_img_dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        pre_layer_norm: bool = False,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()

        # 0. Handle dimensions
        self.inner_dim = num_attention_heads * attention_head_dim

        assert sample_size is not None, "UniDiffuserModel over patched input must provide sample_size"
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.patch_size = patch_size
        # Assume image is square...
        self.num_patches = (self.sample_size // patch_size) * (self.sample_size // patch_size)

        # 1. Define input layers
        # For now, only support patch input for VAE latent image input
        self.vae_img_in = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
        )
        self.clip_img_in = nn.Linear(clip_img_dim, self.inner_dim)
        self.text_in = nn.Linear(text_dim, self.inner_dim)

        # Timestep embeddings for t_img, t_text
        self.t_img_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.t_img_embed = TimestepEmbedding(
            self.inner_dim,
            4 * self.inner_dim,
            out_dim=self.inner_dim,
        )
        self.t_text_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.t_text_embed = TimestepEmbedding(
            self.inner_dim,
            4 * self.inner_dim,
            out_dim=self.inner_dim,
        )

        # 2. Define transformer blocks
        self.transformer = UTransformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            pre_layer_norm=pre_layer_norm,
            norm_elementwise_affine=norm_elementwise_affine,
        )

        # 3. Define output layers
        patch_dim = (patch_size**2) * out_channels
        self.vae_img_out = nn.Linear(self.inner_dim, patch_dim)
        self.clip_img_out = nn.Linear(self.inner_dim, clip_img_dim)
        self.text_out = nn.Linear(self.inner_dim, text_dim)

    def forward(
        self,
        img_vae: torch.FloatTensor,
        img_clip: torch.FloatTensor,
        text: torch.FloatTensor,
        t_img: Union[torch.Tensor, float, int],
        t_text: Union[torch.Tensor, float, int],
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
    ):
        """
        Args:
            img_vae (`torch.FloatTensor` of shape `(batch size, latent channels, height, width)`):
                Latent image representation from the VAE encoder.
            img_clip (`torch.FloatTensor` of shape `(batch size, 1, clip_img_dim)`):
                CLIP-embedded image representation (unsqueezed in the first dimension).
            text (`torch.FloatTensor` of shape `(batch size, seq_len, text_dim)`):
                CLIP-embedded text representation.
            t_img (`torch.long` or `float` or `int`):
                Current denoising step for the image.
            t_text (`torch.long` or `float` or `int`):
                Current denoising step for the text.
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        batch_size = img_vae.shape[0]

        # 1. Input
        # 1.1. Map inputs to shape (B, N, inner_dim)
        vae_hidden_states = self.vae_img_in(img_vae)
        clip_hidden_states = self.clip_img_in(img_clip)
        text_hidden_states = self.text_in(text)

        # print(f"VAE embedding shape: {vae_hidden_states.shape}")
        # print(f"CLIP embedding shape: {clip_hidden_states.shape}")
        # print(f"Prompt embedding shape: {text_hidden_states.shape}")

        num_text_tokens, num_img_tokens = text_hidden_states.size(1), vae_hidden_states.size(1)

        # 1.2. Encode image timesteps
        if not torch.is_tensor(t_img):
            t_img = torch.tensor([t_img], dtype=torch.long, device=vae_hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t_img = t_img * torch.ones(batch_size, dtype=t_img.dtype, device=t_img.device)

        t_img_token = self.t_img_proj(t_img)
        # t_img_token does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        t_img_token = t_img_token.to(dtype=self.dtype)
        t_img_token = self.t_img_embed(t_img_token)
        t_img_token = t_img_token.unsqueeze(dim=1)
        # print(f"Image timestep token shape: {t_img_token.shape}")

        # 1.3. Encode text timesteps
        if not torch.is_tensor(t_text):
            t_text = torch.tensor([t_text], dtype=torch.long, device=vae_hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t_text = t_text * torch.ones(batch_size, dtype=t_text.dtype, device=t_text.device)

        t_text_token = self.t_text_proj(t_text)
        # t_text_token does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        t_text_token = t_text_token.to(dtype=self.dtype)
        t_text_token = self.t_text_embed(t_text_token)
        t_text_token = t_text_token.unsqueeze(dim=1)
        # print(f"Text timestep token shape: {t_text_token.shape}")

        # 1.4. Concatenate all of the embeddings together.
        hidden_states = torch.cat(
            [t_img_token, t_text_token, text_hidden_states, clip_hidden_states, vae_hidden_states], dim=1
        )

        # 2. Blocks
        hidden_states = self.transformer(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            hidden_states_is_embedding=True,
            unpatchify=False,
        )[0]

        # print(f"Transformer output shape: {hidden_states.shape}")

        # 3. Output
        # Split out the predicted noise representation.
        t_img_token_out, t_text_token_out, text_out, img_clip_out, img_vae_out = hidden_states.split(
            (1, 1, num_text_tokens, 1, num_img_tokens), dim=1
        )

        # print(F"img vae transformer output shape: {img_vae_out.shape}")

        img_vae_out = self.vae_img_out(img_vae_out)
        # print(f"img_vae_out shape: {img_vae_out.shape}")
        # unpatchify
        height = width = int(img_vae_out.shape[1] ** 0.5)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        img_vae_out = torch.einsum("nhwpqc->nchpwq", img_vae_out)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        img_clip_out = self.clip_img_out(img_clip_out)

        text_out = self.text_out(text_out)

        return img_vae_out, img_clip_out, text_out
