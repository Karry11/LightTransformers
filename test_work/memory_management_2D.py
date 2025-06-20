import torch

class TokenBuffer2D:
    def __init__(self, total_bytes: int, dtype=torch.int32, device='cuda'):
        """
        total_bytes: 总共分配的显存字节数 (2 * 1024**3 for 2 GiB)
        dtype:       存储的元素类型 (这里用 int32 存 token id)
        device:      'cuda' 或者具体设备名
        """
        self.dtype = dtype
        self.device = device

        # 计算可存元素总数
        elem_size = torch.empty((), dtype=dtype).element_size()
        total_elems = total_bytes // elem_size

        # 每个“slot”存一个 token 的向量，长度 1024
        # 每行放 4 个 slot
        elems_per_row = 4 * 1024
        self.num_rows = total_elems // elems_per_row

        # 一次性在 GPU 上分配：rows × 4 cols × 1024 向量
        self.buffer = torch.empty(
            (self.num_rows, 4, 1024),
            dtype=dtype,
            device=device
        )
        # self.buffer.zero_()
        # 写入指针：row, col
        self.row = 0
        self.col = 0

    def write(self, token_id: int):
        """将 token_id 扩展成 [token_id, …] 长度 1024 的 vector 写入当前位置"""
        if self.row >= self.num_rows:
            raise RuntimeError("Buffer overflow: no more rows available")

        # 填充当前 slot
        self.buffer[self.row, self.col].fill_(token_id)

        # 记录写入位置，方便外部打印
        written_pos = (self.row, self.col)

        # 推进指针
        self.col += 1
        if self.col >= 4:
            self.col = 0
            self.row += 1

        return written_pos

    def find(self, token_id: int):
        """查找 token_id 存放的第一个位置，返回 (row, col) 或 None"""
        # 只需检查每个 slot 的第一个元素是否等于 token_id
        # 因为我们填充时全向量都是同一个值
        # buffer[:, :, 0] 形状 [num_rows, 4]
        mask = (self.buffer[:, :, 0] == token_id).cpu()
        nz = mask.nonzero(as_tuple=False)
        if nz.numel() == 0:
            return None
        row, col = nz[0].tolist()
        return (row, col)

if __name__ == "__main__":
    # 1) 初始化一个 2 GiB 的 buffer
    total_bytes = 2 * 1024**3
    buf = TokenBuffer2D(total_bytes, dtype=torch.int32, device='cuda')

    # 2) CLI 循环
    print("输入格式：'<token_id> write' 或 '<token_id> find'，Ctrl+C 退出")
    while True:
        cmd = input("> ").strip()
        parts = cmd.split()
        if len(parts) != 2 or parts[1] not in {"write", "find"}:
            print("无效命令，示例：'101 write' 或 '101 find'")
            continue

        token_id = int(parts[0])
        op = parts[1]

        if op == "write":
            try:
                row, col = buf.write(token_id)
                print(f"Written token {token_id} at row {row}, col {col}")
            except RuntimeError as e:
                print("Error:", e)
                break

        else:  # op == "find"
            pos = buf.find(token_id)
            if pos is None:
                print("No write")
            else:
                print(f"Token {token_id} found at row {pos[0]}, col {pos[1]}")
