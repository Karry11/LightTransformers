from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ─── 1. 加载模型与 tokenizer ────────────────────────────────────────────────
model_path = "/mnt/s/NLP/LocalModel/qwen2-0.5b/"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 确保开启缓存
model.config.use_cache = True

# 获取模型所在设备（CPU/GPU）
device = next(model.parameters()).device

# ─── 2. Prefill：对 prompt 进行前缀计算，得到初始 KV-cache ────────────────
prompt = "你好大海，我是melo"

# 把 prompt 编码为 tensor，并移动到模型设备
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    prefill_outputs = model(
        **inputs,
        use_cache=True,
        return_dict=True
    )
    past_key_values = prefill_outputs.past_key_values  # 初始 KV-cache

# ─── 3. 手动循环生成新 token ───────────────────────────────────────────────
max_new_tokens = 20
generated_tokens = []

for step in range(max_new_tokens):
    # 取上一步的 token 作为输入
    if step == 0:
        # 第一次，用 prompt 的最后一个 token
        next_input_ids = inputs["input_ids"][:, -1].unsqueeze(-1)  # shape [1,1]
    else:
        # 后续，用上一次生成的 token
        next_input_ids = torch.tensor(
            [[generated_tokens[-1]]],
            device=device,
            dtype=torch.long
        )

    # 前向计算，传入上次的 past_key_values
    with torch.no_grad():
        outputs = model(
            input_ids=next_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        logits = outputs.logits               # [1, 1, vocab_size]
        past_key_values = outputs.past_key_values  # 更新 KV-cache

    # 贪心解码：选最大概率的 token
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated_tokens.append(next_token)

# ─── 4. 合并并解码完整序列 ────────────────────────────────────────────────
all_ids = torch.cat(
    [inputs["input_ids"][0], torch.tensor(generated_tokens, device=device)],
    dim=0
)
result = tokenizer.decode(all_ids, skip_special_tokens=True)

print("Generated token IDs:", generated_tokens)
print("Full generated text: ", result)
