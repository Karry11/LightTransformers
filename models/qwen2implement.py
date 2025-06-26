from transformers.models.qwen2.modeling_qwen2 import *
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
        orig_shape = hidden_states.shape
        batch, seq_len, hidden_size = orig_shape
        x = hidden_states.to(torch.float32).view(-1, hidden_size)

        y = rmsnorm_triton(x, self.weight.to(torch.float32), self.variance_epsilon)

        y = y.view(batch, seq_len, hidden_size).to(input_dtype)
        return y

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"


class qwen2modelimplement(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2ModelTriron(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

class Qwen2ModelTriron(Qwen2Model):
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

