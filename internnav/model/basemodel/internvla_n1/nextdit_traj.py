# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
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

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import LuminaFeedForward
from diffusers.models.attention_processor import Attention, LuminaAttnProcessor2_0
from diffusers.models.embeddings import (
    LuminaCombinedTimestepCaptionEmbedding,
    LuminaPatchEmbed,
    PixArtAlphaTextProjection,
    apply_rotary_emb,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    LuminaLayerNormContinuous,
    LuminaRMSNormZero,
    RMSNorm,
)
from diffusers.utils import is_torch_version, logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _ensure_cond_cache_metadata(cond_cache: Optional[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if cond_cache is None:
        return None, None

    stats = cond_cache.setdefault("_stats", {})
    stats.setdefault("hits", 0)
    stats.setdefault("misses", 0)
    stats.setdefault("saved_ms_total", 0.0)
    stats.setdefault("crossattn_kv_hits", 0)
    stats.setdefault("crossattn_kv_misses", 0)
    stats.setdefault("crossattn_kv_saved_ms_total", 0.0)

    timings = cond_cache.setdefault("_timings", {})
    timings.setdefault("projected_encoder_hidden_states_ms", None)
    timings.setdefault("layer_normed_encoder_hidden_states_ms", [])
    timings.setdefault("layer_crossattn_kv_ms", [])
    return stats, timings


def _record_cond_cache_access(
    cond_cache: Optional[Dict[str, Any]],
    *,
    hit: bool,
    saved_ms: float = 0.0,
    kv: bool = False,
) -> None:
    stats, _ = _ensure_cond_cache_metadata(cond_cache)
    if stats is None:
        return

    if kv:
        key_hits = "crossattn_kv_hits"
        key_misses = "crossattn_kv_misses"
        key_saved = "crossattn_kv_saved_ms_total"
    else:
        key_hits = "hits"
        key_misses = "misses"
        key_saved = "saved_ms_total"

    if hit:
        stats[key_hits] += 1
        stats[key_saved] += float(saved_ms)
    else:
        stats[key_misses] += 1


def _build_crossattn_kv_from_encoder_hidden_states(
    attn: Attention,
    encoder_hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Exact encoder-side cache entry: K/V depend only on the per-layer normalized
    # encoder condition and can be safely reused across diffusion steps.
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    if attn.norm_k is not None:
        key = attn.norm_k(key)

    batch_size = encoder_hidden_states.shape[0]
    head_dim = attn.inner_dim // attn.heads
    kv_heads = key.shape[-1] // head_dim

    key = key.view(batch_size, -1, kv_heads, head_dim)
    value = value.view(batch_size, -1, kv_heads, head_dim)

    n_rep = attn.heads // kv_heads
    if n_rep >= 1:
        key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

    return key.transpose(1, 2), value.transpose(1, 2)


def _maybe_get_cached_crossattn_kv(
    attn: Attention,
    layer_idx: int,
    total_layers: int,
    encoder_hidden_states: torch.Tensor,
    cond_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, timings = _ensure_cond_cache_metadata(cond_cache)
    use_cache = bool(cond_cache is not None and cond_cache.get("crossattn_kv_cache_enabled"))

    if not use_cache:
        return _build_crossattn_kv_from_encoder_hidden_states(attn, encoder_hidden_states)

    key_cache = cond_cache.get("layer_crossattn_k_cache")
    value_cache = cond_cache.get("layer_crossattn_v_cache")
    if key_cache is None or value_cache is None:
        key_cache = [None] * total_layers
        value_cache = [None] * total_layers
        cond_cache["layer_crossattn_k_cache"] = key_cache
        cond_cache["layer_crossattn_v_cache"] = value_cache

    if layer_idx < len(key_cache) and key_cache[layer_idx] is not None and value_cache[layer_idx] is not None:
        saved_ms = 0.0
        if timings is not None:
            layer_timings = timings.get("layer_crossattn_kv_ms") or []
            if layer_idx < len(layer_timings):
                saved_ms = float(layer_timings[layer_idx] or 0.0)
        _record_cond_cache_access(cond_cache, hit=True, saved_ms=saved_ms, kv=True)
        return key_cache[layer_idx], value_cache[layer_idx]

    start = time.perf_counter()
    key, value = _build_crossattn_kv_from_encoder_hidden_states(attn, encoder_hidden_states)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    key_cache[layer_idx] = key
    value_cache[layer_idx] = value

    if timings is not None:
        layer_timings = timings.setdefault("layer_crossattn_kv_ms", [])
        if len(layer_timings) < total_layers:
            layer_timings.extend([None] * (total_layers - len(layer_timings)))
        if layer_timings[layer_idx] is None:
            layer_timings[layer_idx] = elapsed_ms

    _record_cond_cache_access(cond_cache, hit=False, kv=True)
    return key, value


def _apply_cross_attention_with_cached_encoder_kv(
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_rotary_emb: Optional[torch.Tensor],
    cond_cache: Optional[Dict[str, Any]],
    layer_idx: int,
    total_layers: int,
) -> torch.Tensor:
    query = attn.to_q(hidden_states)
    query_dim = query.shape[-1]
    head_dim = query_dim // attn.heads
    dtype = query.dtype

    if attn.norm_q is not None:
        query = attn.norm_q(query)

    query = query.view(hidden_states.shape[0], -1, attn.heads, head_dim)
    if query_rotary_emb is not None:
        query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
    query = query.to(dtype)

    key, value = _maybe_get_cached_crossattn_kv(
        attn,
        layer_idx,
        total_layers,
        encoder_hidden_states,
        cond_cache=cond_cache,
    )
    key = key.to(dtype)

    if attention_mask is not None:
        attention_mask = attention_mask.bool().view(hidden_states.shape[0], 1, 1, -1)
        attention_mask = attention_mask.expand(-1, attn.heads, hidden_states.shape[1], -1)

    query = query.transpose(1, 2)
    hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=None)
    return hidden_states.transpose(1, 2).to(dtype)


class LuminaNextDiTBlock(nn.Module):
    """
    A LuminaNextDiTBlock for LuminaNextDiT2DModel.

    Parameters:
        dim (`int`): Embedding dimension of the input features.
        num_attention_heads (`int`): Number of attention heads.
        num_kv_heads (`int`):
            Number of attention heads in key and value features (if using GQA), or set to None for the same as query.
        multiple_of (`int`): The number of multiple of ffn layer.
        ffn_dim_multiplier (`float`): The multipier factor of ffn layer dimension.
        norm_eps (`float`): The eps for norm layer.
        qk_norm (`bool`): normalization for query and key.
        cross_attention_dim (`int`): Cross attention embedding dimension of the input text prompt hidden_states.
        norm_elementwise_affine (`bool`, *optional*, defaults to True),
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        cross_attention_dim: int,
        norm_elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads

        self.gate = nn.Parameter(torch.zeros([num_attention_heads]))

        # Self-attention
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="layer_norm_across_heads" if qk_norm else None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.attn1.to_out = nn.Identity()

        # Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            qk_norm="layer_norm_across_heads" if qk_norm else None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )

        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.norm1 = LuminaRMSNormZero(
            embedding_dim=dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=norm_elementwise_affine,
        )
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm1_context = RMSNorm(cross_attention_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        temb: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        norm_encoder_hidden_states: Optional[torch.Tensor] = None,
        cond_cache: Optional[Dict[str, Any]] = None,
        layer_idx: Optional[int] = None,
        total_layers: Optional[int] = None,
        dit_crossattn_kv_cache_enabled: bool = False,
    ):
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Parameters:
            hidden_states (`torch.Tensor`): The input of hidden_states for LuminaNextDiTBlock.
            attention_mask (`torch.Tensor): The input of hidden_states corresponse attention mask.
            image_rotary_emb (`torch.Tensor`): Precomputed cosine and sine frequencies.
            encoder_hidden_states: (`torch.Tensor`): The hidden_states of text prompt are processed by Gemma encoder.
            encoder_mask (`torch.Tensor`): The hidden_states of text prompt attention mask.
            temb (`torch.Tensor`): Timestep embedding with text prompt embedding.
            cross_attention_kwargs (`Dict[str, Any]`): kwargs for cross attention.
        """
        residual = hidden_states

        # Self-attention
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        self_attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=image_rotary_emb,
            **cross_attention_kwargs,
        )

        # Cross-attention
        if norm_encoder_hidden_states is None:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        use_crossattn_kv_cache = bool(
            dit_crossattn_kv_cache_enabled and cond_cache is not None and layer_idx is not None and total_layers is not None
        )
        if use_crossattn_kv_cache:
            cross_attn_output = _apply_cross_attention_with_cached_encoder_kv(
                self.attn2,
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=encoder_mask,
                query_rotary_emb=image_rotary_emb,
                cond_cache=cond_cache,
                layer_idx=layer_idx,
                total_layers=total_layers,
            )
        else:
            cross_attn_output = self.attn2(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=encoder_mask,
                query_rotary_emb=image_rotary_emb,
                key_rotary_emb=None,
                **cross_attention_kwargs,
            )
        cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
        mixed_attn_output = self_attn_output + cross_attn_output
        mixed_attn_output = mixed_attn_output.flatten(-2)
        # linear proj
        hidden_states = self.attn2.to_out[0](mixed_attn_output)

        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(hidden_states)

        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))

        hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)

        return hidden_states


class LuminaNextDiT2DModel(ModelMixin, ConfigMixin):
    """
    LuminaNextDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        in_channels (`int`, *optional*, defaults to 4):
            The number of input channels for the model. Typically, this matches the number of channels in the input
            images.
        hidden_size (`int`, *optional*, defaults to 4096):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        num_layers (`int`, *optional*, default to 32):
            The number of layers in the model. This defines the depth of the neural network.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in each attention layer. This parameter specifies how many separate attention
            mechanisms are used.
        num_kv_heads (`int`, *optional*, defaults to 8):
            The number of key-value heads in the attention mechanism, if different from the number of attention heads.
            If None, it defaults to num_attention_heads.
        multiple_of (`int`, *optional*, defaults to 256):
            A factor that the hidden size should be a multiple of. This can help optimize certain hardware
            configurations.
        ffn_dim_multiplier (`float`, *optional*):
            A multiplier for the dimensionality of the feed-forward network. If None, it uses a default value based on
            the model configuration.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            A small value added to the denominator for numerical stability in normalization layers.
        learn_sigma (`bool`, *optional*, defaults to True):
            Whether the model should learn the sigma parameter, which might be related to uncertainty or variance in
            predictions.
        qk_norm (`bool`, *optional*, defaults to True):
            Indicates if the queries and keys in the attention mechanism should be normalized.
        cross_attention_dim (`int`, *optional*, defaults to 2048):
            The dimensionality of the text embeddings. This parameter defines the size of the text representations used
            in the model.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            A scaling factor applied to certain parameters or layers in the model. This can be used for adjusting the
            overall scale of the model's operations.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["LuminaNextDiTBlock"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: Optional[int] = 2,
        in_channels: Optional[int] = 4,
        hidden_size: Optional[int] = 2304,
        num_layers: Optional[int] = 32,  # 32
        num_attention_heads: Optional[int] = 32,  # 32
        num_kv_heads: Optional[int] = None,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: Optional[float] = 1e-5,
        learn_sigma: Optional[bool] = True,
        qk_norm: Optional[bool] = True,
        cross_attention_dim: Optional[int] = 2048,
        scaling_factor: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling_factor = scaling_factor
        self.gradient_checkpointing = False

        self.caption_projection = PixArtAlphaTextProjection(in_features=cross_attention_dim, hidden_size=hidden_size)
        self.patch_embedder = LuminaPatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True
        )

        self.time_caption_embed = LuminaCombinedTimestepCaptionEmbedding(
            hidden_size=min(hidden_size, 1024), cross_attention_dim=hidden_size
        )

        self.layers = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    hidden_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )
        # self.final_layer = LuminaFinalLayer(hidden_size, patch_size, self.out_channels)

        assert (hidden_size // num_attention_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _get_projected_encoder_hidden_states(
        self,
        encoder_hidden_states: torch.Tensor,
        cond_cache: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Exact cache entry: depends only on encoder-side condition tokens.
        _, timings = _ensure_cond_cache_metadata(cond_cache)
        if cond_cache is not None:
            cached = cond_cache.get("projected_encoder_hidden_states")
            if cached is not None:
                saved_ms = float((timings or {}).get("projected_encoder_hidden_states_ms") or 0.0)
                _record_cond_cache_access(cond_cache, hit=True, saved_ms=saved_ms)
                return cached

        start = time.perf_counter()
        projected_encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if cond_cache is not None:
            cond_cache["projected_encoder_hidden_states"] = projected_encoder_hidden_states
            if timings is not None and timings.get("projected_encoder_hidden_states_ms") is None:
                timings["projected_encoder_hidden_states_ms"] = elapsed_ms
            _record_cond_cache_access(cond_cache, hit=False)

        return projected_encoder_hidden_states

    def _get_layer_normed_encoder_hidden_states(
        self,
        layer: LuminaNextDiTBlock,
        layer_idx: int,
        encoder_hidden_states: torch.Tensor,
        cond_cache: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Exact cache entry: per-layer encoder-side norm, independent of latents/timestep.
        _, timings = _ensure_cond_cache_metadata(cond_cache)
        layer_cache = None if cond_cache is None else cond_cache.get("layer_normed_encoder_hidden_states")
        if layer_cache is not None and layer_idx < len(layer_cache) and layer_cache[layer_idx] is not None:
            saved_ms = 0.0
            if timings is not None:
                layer_timings = timings.get("layer_normed_encoder_hidden_states_ms") or []
                if layer_idx < len(layer_timings):
                    saved_ms = float(layer_timings[layer_idx] or 0.0)
            _record_cond_cache_access(cond_cache, hit=True, saved_ms=saved_ms)
            return layer_cache[layer_idx]

        start = time.perf_counter()
        normed_encoder_hidden_states = layer.norm1_context(encoder_hidden_states)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if cond_cache is not None:
            if layer_cache is None:
                layer_cache = [None] * len(self.layers)
                cond_cache["layer_normed_encoder_hidden_states"] = layer_cache
            layer_cache[layer_idx] = normed_encoder_hidden_states

            if timings is not None:
                layer_timings = timings.setdefault("layer_normed_encoder_hidden_states_ms", [])
                if len(layer_timings) < len(self.layers):
                    layer_timings.extend([None] * (len(self.layers) - len(layer_timings)))
                if layer_timings[layer_idx] is None:
                    layer_timings[layer_idx] = elapsed_ms
            _record_cond_cache_access(cond_cache, hit=False)

        return normed_encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any] = None,
        cond_cache: Optional[Dict[str, Any]] = None,
        dit_crossattn_kv_cache_enabled: bool = False,
        return_dict=True,
    ) -> torch.Tensor:
        """
        Forward pass of LuminaNextDiT.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
        """

        mask = torch.ones(
            hidden_states.shape[0], hidden_states.shape[1], dtype=torch.int32, device=hidden_states.device
        )
        cross_attention_kwargs = cross_attention_kwargs or {}
        if dit_crossattn_kv_cache_enabled and cond_cache is None:
            raise ValueError("dit_crossattn_kv_cache_enabled requires cond_cache. generate_traj() should supply it.")
        encoder_hidden_states = self._get_projected_encoder_hidden_states(
            encoder_hidden_states,
            cond_cache=cond_cache,
        )
        temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)

        encoder_mask = encoder_mask.bool()

        for layer_idx, layer in enumerate(self.layers):
            norm_encoder_hidden_states = self._get_layer_normed_encoder_hidden_states(
                layer,
                layer_idx,
                encoder_hidden_states,
                cond_cache=cond_cache,
            )
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, current_layer_idx: int):
                    def custom_forward(
                        hidden_states,
                        attention_mask,
                        image_rotary_emb,
                        encoder_hidden_states,
                        encoder_mask,
                        temb,
                        norm_encoder_hidden_states,
                    ):
                        return module(
                            hidden_states,
                            attention_mask,
                            image_rotary_emb,
                            encoder_hidden_states,
                            encoder_mask,
                            temb=temb,
                            cross_attention_kwargs=cross_attention_kwargs,
                            norm_encoder_hidden_states=norm_encoder_hidden_states,
                            cond_cache=cond_cache,
                            layer_idx=current_layer_idx,
                            total_layers=len(self.layers),
                            dit_crossattn_kv_cache_enabled=dit_crossattn_kv_cache_enabled,
                        )

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer, layer_idx),
                    hidden_states,
                    mask,
                    image_rotary_emb,
                    encoder_hidden_states,
                    encoder_mask,
                    temb,
                    norm_encoder_hidden_states,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    mask,
                    image_rotary_emb,
                    encoder_hidden_states,
                    encoder_mask,
                    temb=temb,
                    cross_attention_kwargs=cross_attention_kwargs,
                    norm_encoder_hidden_states=norm_encoder_hidden_states,
                    cond_cache=cond_cache,
                    layer_idx=layer_idx,
                    total_layers=len(self.layers),
                    dit_crossattn_kv_cache_enabled=dit_crossattn_kv_cache_enabled,
                )

        hidden_states = self.norm_out(hidden_states, temb)

        output = hidden_states
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
