# import torch
# import torch.nn as nn
#
# class Qwen2RMSNorm(nn.Module):
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
import torch
import triton
import triton.language as tl
import time
@triton.jit
def rmsnorm_triton_kernel_calculation(
        X,  # (batch_size, hidden_size)
        W,  # (hidden_size,)
        Y,  # output buffer (batch_size, hidden_size)
        stride_xm, stride_xn,
        stride_w,
        stride_ym, stride_yn,
        eps,
        BLOCK: tl.constexpr
):
    batch_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    x = tl.load(X + batch_id * stride_xm + offset * stride_xn)
    w = tl.load(W + offset * stride_w)
    var = tl.sum(x * x, axis=0) / BLOCK
    inv = tl.rsqrt(var + eps)
    y = x * inv
    y *= w
    tl.store(Y + batch_id * stride_ym + offset * stride_yn, y)



# Triton kernel for RMSNorm
@triton.jit
def rmsnorm_triton_kernel(
        X,  # (batch_size, hidden_size)
        W,  # (hidden_size,)
        Y,  # output buffer (batch_size, hidden_size)
        stride_xm, stride_xn,
        stride_w,
        stride_ym, stride_yn,
        eps,
        BLOCK: tl.constexpr
):
    batch_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # Load input and weight
    x = tl.load(X + batch_idx * stride_xm + offs * stride_xn)
    w = tl.load(W + offs * stride_w)

    # Compute variance = mean(x^2)
    var = tl.sum(x * x, axis=0) / BLOCK

    # Normalize
    inv = tl.rsqrt(var + eps)
    y = x * inv

    # Apply weight
    y = y * w

    # Store result
    tl.store(Y + batch_idx * stride_ym + offs * stride_yn, y)


# Triton wrapper
def rmsnorm_triton(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    B, N = x.shape
    y = torch.empty_like(x)
    rmsnorm_triton_kernel[(B,)](
        x, w, y,
        x.stride(0), x.stride(1),
        w.stride(0),
        y.stride(0), y.stride(1),
        eps,
        BLOCK=N
    )
    return y

def rmsnorn_triton_calculation(x: torch.Tensor, w: torch.Tensor, eps:float = 1e-6) -> torch:
    B, N = x.shape
    y = torch.empty_like(x)
    rmsnorm_triton_kernel_calculation[(B,)](
        x, w, y,
        x.stride(0), x.stride(1),
        w.stride(0),
        y.stride(0), y.stride(1),
        eps,
        BLOCK=N
    )
    return y



# Pure PyTorch reference
def rmsnorm_pytorch(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    dtype = x.dtype
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(-1, keepdim=True)
    y = x_fp32 * (var + eps).rsqrt()
    y = y * w
    return y.to(dtype)


# Benchmark comparison
if __name__ == "__main__":
    device = 'cuda'
    B, N = 32, 4096
    x = torch.randn(B, N, device=device, dtype=torch.float16)
    w = torch.ones(N, device=device, dtype=torch.float16)

    # Warm-up
    for _ in range(5):
        _ = rmsnorm_pytorch(x, w)
        _ = rmsnorm_triton(x, w)


    # Timing helper
    def time_fn(fn, *args):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)


    pt_time = time_fn(rmsnorm_pytorch, x, w)
    triton_time = time_fn(rmsnorm_triton, x, w)

    print(f"PyTorch RMSNorm: {pt_time:.3f} ms")
    print(f"Triton  RMSNorm: {triton_time:.3f} ms")
    print(f"Speedup: {pt_time / triton_time:.2f}x")
    flag = rmsnorm_triton(x,w) == rmsnorm_pytorch(x,w)
    print(flag)

