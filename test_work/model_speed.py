import torch
import triton
import triton.language as tl
import time


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
    print(rmsnorm_triton(x, w))
    print("123")
    print(rmsnorm_pytorch(x, w))
    print(torch.allclose(rmsnorm_triton(x, w), rmsnorm_pytorch(x, w),))

