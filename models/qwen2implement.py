from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
from torch import nn
import triton

import torch
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import *
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import triton
import triton.language as tl
from math import prod

# ─── Triton kernel ────────────────────────────────────────────────────────────
@triton.jit
def rmsnorm_triton_kernel(
    X,         # (B, BLOCK) float32
    W,         # (BLOCK,)   float32
    Y,         # (B, BLOCK) float32
    stride_xm, stride_xn,
    stride_w,
    stride_ym, stride_yn,
    eps,
    hidden_size: tl.constexpr,  # real hidden_size
    BLOCK: tl.constexpr         # must be power-of-two (1024)
):
    batch_idx = tl.program_id(0)
    offs      = tl.arange(0, BLOCK)
    mask      = offs < hidden_size

    # load a BLOCK-sized slice (masked beyond hidden_size)
    x = tl.load(X + batch_idx * stride_xm + offs * stride_xn, mask=mask, other=0.0)
    w = tl.load(W + offs * stride_w,                         mask=mask, other=0.0)

    # compute variance = mean(x^2) over real hidden_size
    sum_x2 = tl.sum(x * x, axis=0)
    var    = sum_x2 / hidden_size
    inv    = tl.rsqrt(var + eps)

    # normalize + apply weight
    y = x * inv * w

    # store back (masked)
    tl.store(Y + batch_idx * stride_ym + offs * stride_yn, y, mask=mask)

# ─── Triton wrapper ───────────────────────────────────────────────────────────
def rmsnorm_triton(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """
    x: (..., hidden_size) float32
    w: (hidden_size,)        float32
    returns y: same shape & dtype as x
    """
    orig_shape = x.shape
    *front, N = orig_shape
    B = prod(front) or 1

    x_flat = x.contiguous().view(B, N).float()
    w_flat = w.contiguous().view(N).float()
    y_flat = torch.empty_like(x_flat)

    BLOCK = 1024
    assert N <= BLOCK, f"hidden_size ({N}) must be ≤ BLOCK ({BLOCK})"
    grid = (B,)

    rmsnorm_triton_kernel[grid](
        x_flat, w_flat, y_flat,
        x_flat.stride(0), x_flat.stride(1),
        w_flat.stride(0),
        y_flat.stride(0), y_flat.stride(1),
        eps, N, BLOCK
    )
    return y_flat.view(*orig_shape)

# ─── PyTorch Module ───────────────────────────────────────────────────────────
class Qwen2RMSNormTriton(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        """
        Triton-accelerated RMSNorm, equivalent to T5LayerNorm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, hidden_size), float16 or float32
        returns same shape & dtype
        """
        input_dtype = hidden_states.dtype
        # convert to float32 and flatten leading dims
        orig_shape = hidden_states.shape
        batch, seq_len, hidden_size = orig_shape
        x = hidden_states.to(torch.float32).view(-1, hidden_size)

        # Triton RMSNorm
        y = rmsnorm_triton(x, self.weight.to(torch.float32), self.variance_epsilon)

        # reshape back and convert dtype
        y = y.view(batch, seq_len, hidden_size).to(input_dtype)
        return y

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"

# import triton
# import triton.language as tl
#
# # ─── Triton kernel ────────────────────────────────────────────────────────────
# @triton.jit
# def rmsnorm_triton_kernel(
#     X,       # (batch_size, hidden_size)
#     W,       # (hidden_size,)
#     Y,       # output buffer (batch_size, hidden_size)
#     stride_xm, stride_xn,
#     stride_w,
#     stride_ym, stride_yn,
#     eps,
#     BLOCK: tl.constexpr
# ):
#     batch_idx = tl.program_id(0)
#     offs = tl.arange(0, BLOCK)
#
#     # Load input and weight
#     x = tl.load(X + batch_idx * stride_xm + offs * stride_xn)
#     w = tl.load(W + offs * stride_w)
#
#     # Compute variance = mean(x^2)
#     var = tl.sum(x * x, axis=0) / BLOCK
#
#     # Normalize
#     inv = tl.rsqrt(var + eps)
#     y = x * inv
#
#     # Apply weight
#     y = y * w
#
#     # Store result
#     tl.store(Y + batch_idx * stride_ym + offs * stride_yn, y)
#
# # Triton wrapper
# def rmsnorm_triton(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
#     """
#     x: (B, N) float32 tensor
#     w: (N,)      float32 tensor
#     returns y:  (B, N) float32 tensor
#     """
#     B, N = x.shape
#     y = torch.empty_like(x)
#     # launch one program per batch row
#     rmsnorm_triton_kernel[(B,)](
#         x, w, y,
#         x.stride(0), x.stride(1),
#         w.stride(0),
#         y.stride(0), y.stride(1),
#         eps,
#         BLOCK=N
#     )
#     return y
#
# # ─── PyTorch Module ───────────────────────────────────────────────────────────
# class Qwen2RMSNormTriton(nn.Module):
#     def __init__(self, hidden_size, eps: float = 1e-6):
#         """
#         Triton-accelerated RMSNorm, equivalent to T5LayerNorm
#         """
#         super().__init__()
#         # 注意：权重保留为 float32，或者在 forward 里转回 float32
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         input_dtype = hidden_states.dtype
#         orig_shape = hidden_states.shape  # 例： (batch, seq_len, hidden_size)
#         hidden_size = orig_shape[-1]
#         print("orig shape: ", orig_shape)
#
#         # 1. 转为 float32 并 flatten 到 (B, N)
#         x_flat = hidden_states.to(torch.float32).view(-1, hidden_size)
#
#         # 2. 调用 Triton kernel，得到同样 shape 的输出
#         y_flat = rmsnorm_triton(x_flat, self.weight.to(torch.float32), self.variance_epsilon)
#
#         # 3. 恢复原始维度，并转回原 dtype
#         y = y_flat.view(*orig_shape).to(input_dtype)
#         return y
#
#     def extra_repr(self):
#         return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"

# class Qwen2RMSNormTriton(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Qwen2RMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#
#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)
#
#     def extra_repr(self):
#         return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class qwen2modelimplement(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2ModelTriron(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class Qwen2ModelTriron(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNormTriton(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

# class Qwen2ModelImplement(Qwen2PreTrainedModel):
#     """
#     Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]
#
#     Args:
#         config: Qwen2Config
#     """
#
#     def __init__(self, config: Qwen2Config):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size
#
#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.layers = nn.ModuleList(
#             [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = Qwen2RotaryEmbedding(config=config)
#         self.gradient_checkpointing = False
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def get_input_embeddings(self):
#         return self.embed_tokens
#
#     def set_input_embeddings(self, value):
#         self.embed_tokens = value
#
#     @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
#
#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             use_cache = False
#
#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)
#
#         if use_cache and past_key_values is None:
#             past_key_values = DynamicCache()
#
#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )
#
#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)
#
#         causal_mask = self._update_causal_mask(
#             attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
#         )
#
#         hidden_states = inputs_embeds
#
#         # create position embeddings to be shared across the decoder layers
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)
#
#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#
#         for decoder_layer in self.layers[: self.config.num_hidden_layers]:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
#
#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     causal_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                     cache_position,
#                     position_embeddings,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=causal_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                     cache_position=cache_position,
#                     position_embeddings=position_embeddings,
#                     **flash_attn_kwargs,
#                 )
#
#             hidden_states = layer_outputs[0]
#
#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)
#
#         hidden_states = self.norm(hidden_states)
#
#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)
#
#         output = BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=past_key_values if use_cache else None,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )
#         return output if return_dict else output.to_tuple()
#
#     def _update_causal_mask(
#         self,
#         attention_mask: torch.Tensor,
#         input_tensor: torch.Tensor,
#         cache_position: torch.Tensor,
#         past_key_values: Cache,
#         output_attentions: bool,
#     ):
#         if self.config._attn_implementation == "flash_attention_2":
#             if attention_mask is not None and (attention_mask == 0.0).any():
#                 return attention_mask
#             return None
#
#         # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
#         # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
#         # to infer the attention mask.
#         past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#         using_static_cache = isinstance(past_key_values, StaticCache)
#
#         # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
#         if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
#             if AttentionMaskConverter._ignore_causal_mask_sdpa(
#                 attention_mask,
#                 inputs_embeds=input_tensor,
#                 past_key_values_length=past_seen_tokens,
#                 is_training=self.training,
#             ):
#                 return None
#
#         dtype, device = input_tensor.dtype, input_tensor.device
#         sequence_length = input_tensor.shape[1]
#         if using_static_cache:
#             target_length = past_key_values.get_max_cache_shape()
#         else:
#             target_length = (
#                 attention_mask.shape[-1]
#                 if isinstance(attention_mask, torch.Tensor)
#                 else past_seen_tokens + sequence_length + 1
#             )
#
#         # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
#         causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
#             attention_mask,
#             sequence_length=sequence_length,
#             target_length=target_length,
#             dtype=dtype,
#             device=device,
#             cache_position=cache_position,
#             batch_size=input_tensor.shape[0],
#         )
#
#         if (
#             self.config._attn_implementation == "sdpa"
#             and attention_mask is not None
#             and attention_mask.device.type == "cuda"
#             and not output_attentions
#         ):
#             # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
#             # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
#             # Details: https://github.com/pytorch/pytorch/issues/110213
#             min_dtype = torch.finfo(dtype).min
#             causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
#
#         return causal_mask
#
#     @staticmethod
#     def _prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask: torch.Tensor,
#         sequence_length: int,
#         target_length: int,
#         dtype: torch.dtype,
#         device: torch.device,
#         cache_position: torch.Tensor,
#         batch_size: int,
#         **kwargs,
#     ):
#         """
#         Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
#         `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
#
#         Args:
#             attention_mask (`torch.Tensor`):
#                 A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
#                 `(batch_size, 1, query_length, key_value_length)`.
#             sequence_length (`int`):
#                 The sequence length being processed.
#             target_length (`int`):
#                 The target length: when generating with static cache, the mask should be as long as the static cache,
#                 to account for the 0 padding, the part of the cache that is not filled yet.
#             dtype (`torch.dtype`):
#                 The dtype to use for the 4D attention mask.
#             device (`torch.device`):
#                 The device to plcae the 4D attention mask on.
#             cache_position (`torch.Tensor`):
#                 Indices depicting the position of the input sequence tokens in the sequence.
#             batch_size (`torch.Tensor`):
#                 Batch size.
#         """
#         if attention_mask is not None and attention_mask.dim() == 4:
#             # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
#             causal_mask = attention_mask
#         else:
#             min_dtype = torch.finfo(dtype).min
#             causal_mask = torch.full(
#                 (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
#             )
#             if sequence_length != 1:
#                 causal_mask = torch.triu(causal_mask, diagonal=1)
#             causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
#             causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
#             if attention_mask is not None:
#                 causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
#                 mask_length = attention_mask.shape[-1]
#                 padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
#                 padding_mask = padding_mask == 0
#                 causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
#                     padding_mask, min_dtype
#                 )
#
#         return causal_mask

