from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# ─── 1. 定义一个子类，继承原模型并打印 KV‐cache ────────────────────────────
#    注意：我们先用 from_pretrained 加载一次模型来拿到它的真实类（trust_remote_code 下）
BaseModelClass = AutoModelForCausalLM.from_pretrained(
    "/mnt/s/NLP/LocalModel/qwen2-0.5b/",
    trust_remote_code=False
).__class__

class Qwen2WithKV(BaseModelClass):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs
    ):
        # 调用父类 forward
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        # 如果返回了 KV‐cache，就打印每层的 shape
        pkv = getattr(outputs, "past_key_values", None)
        if pkv is not None:
            print(f"[DEBUG] —— 得到 {len(pkv)} 层 KV‐cache ——")
            for i, (k, v) in enumerate(pkv):
                # k, v 的 shape 通常是 (batch_size, num_heads, seq_len, head_dim)
                print(f"  layer {i:2d}: k {tuple(k.shape)},  v {tuple(v.shape)}")
        return outputs

# ─── 2. 用子类来加载真正的模型 ────────────────────────────────────────────────
model_path = "/mnt/s/NLP/LocalModel/qwen2-0.5b/"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
model = Qwen2WithKV.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = True
device = next(model.parameters()).device

# ─── 3. Prefill + 手动循环生成（同上，只是输出时会触发打印） ───────────────────
prompt = "你好大海，我是melo"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    # 第一次调用 forward 时，会打印出初始 KV‐cache
    prefill_out = model(
        **inputs,
        use_cache=True,
        return_dict=True
    )
    past_key_values = prefill_out.past_key_values

max_new_tokens = 20
generated = []

for step in range(max_new_tokens):
    # 准备下一步输入
    if step == 0:
        next_ids = inputs["input_ids"][:, -1].unsqueeze(-1)
    else:
        next_ids = torch.tensor([[generated[-1]]], device=device)
    with torch.no_grad():
        # 每一步调用都会打印出那一步更新后的 KV‐cache
        out = model(
            input_ids=next_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        logits = out.logits
        past_key_values = out.past_key_values

    nxt = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated.append(nxt)

# 拼回并解码
all_ids = torch.cat([inputs["input_ids"][0], torch.tensor(generated, device=device)], dim=0)
print("最终生成：", tokenizer.decode(all_ids, skip_special_tokens=True))
