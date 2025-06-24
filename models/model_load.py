from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
import torch
from models.qwen2implement import qwen2modelimplement
def lazy_load_model(config):
    if config.MODEL_NAME == 'qwen2':
        tokenizer = Qwen2Tokenizer.from_pretrained(
            config.MODEL_PATH,
            trust_remote_code=False
        )

        # 加载带有因果语言模型头的 Qwen2 模型
        model = qwen2modelimplement.from_pretrained(
            config.MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False
        )
        return model, tokenizer
