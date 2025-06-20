import torch

class PreallocCudaBuffer:
    def __init__(self, total_bytes: int, dtype=torch.float16, device='cuda'):
        """
        total_bytes:  要分配的总字节数 (2*1024**3 for 2 GiB)
        dtype:        缓冲区里要存的数据类型
        """
        # 单个元素占用的字节数
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        # 可存元素数量
        self.num_elems = total_bytes // elem_bytes
        # 一次性分配
        self.buffer = torch.empty(self.num_elems, dtype=dtype, device=device)
        self.offset = 0
        self.dtype = dtype
        self.device = device

    def write(self, tensor: torch.Tensor):
        """
        将 tensor（任意 shape，dtype 必须匹配）写入缓冲区当前位置，
        并推进 offset。
        """
        assert tensor.dtype == self.dtype, \
            f"Expected dtype={self.dtype}, got {tensor.dtype}"
        flat = tensor.view(-1)
        n = flat.numel()
        if self.offset + n > self.num_elems:
            raise RuntimeError(
                f"Buffer overflow: try to write {n} elems, "
                f"but only {self.num_elems - self.offset} left"
            )
        # 拷贝到大缓冲区
        self.buffer[self.offset : self.offset + n].copy_(flat)
        self.offset += n

    def reset(self):
        """把 offset 清零，复用这块显存"""
        self.offset = 0

# ─── 用法示例 ────────────────────────────────────────────────────────────────

# 1) 一次性申请 2 GiB 的 float16 缓冲区
total_bytes = 2 * 1024**3
buf = PreallocCudaBuffer(total_bytes, dtype=torch.float16, device='cuda')

# 2) 假设每个“token”都对应一个向量 embedding，shape=[batch, hidden_size]
#    这里举例：batch=1, hidden_size=4096
for token_id in [101, 202, 303]:
    # 模拟：把 token_id 转成 embedding（这里只是 demo，用 full of token_id）
    emb = torch.full((1, 4096), float(token_id), dtype=torch.float16, device='cuda')
    buf.write(emb)  # 按顺序写入

# 3) 如果下一个请求需要从头开始写，或者 batch 切换，调用 reset()
buf.reset()
