from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import redis
import json
import uuid
import torch
from common.config import Config
from models.model_load import lazy_load_model
config_ = Config()

redis_pool = redis.ConnectionPool(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=0,
    max_connections=Config.REDIS_MAX_CONNECTIONS,
)

def get_redis():
    return redis.Redis(connection_pool=redis_pool)

# ─── HTTP Server Process ────────────────────────────────────────────────────
def start_http_server():
    app = FastAPI()

    @app.post("/inference")
    async def inference(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            return {"error": "no prompt provided"}

        # 生成唯一 session_id
        session_id = str(uuid.uuid4())
        prompt_key = f"prompt:{session_id}"
        answer_key = f"answer:{session_id}"

        r = get_redis()
        # 清理旧的 answer 列表（如果存在）
        r.delete(answer_key)
        # 将带 session_id 的提示写入统一管道
        payload = json.dumps({"session_id": session_id, "prompt": prompt})
        r.rpush("prompt", payload)

        # 返回包含 session_id 的 header，并流式读取专属 answer 列表
        def event_generator():
            while True:
                _, raw = r.blpop(answer_key)
                token = raw.decode("utf-8")
                yield token
                if token == "[Done]":
                    # 清理 answer 列表
                    r.delete(answer_key)
                    break

        # 在响应 header 中返回 session_id
        headers = {"X-Session-Id": session_id}
        return StreamingResponse(event_generator(), media_type="text/plain", headers=headers)

    uvicorn.run(app, host=Config.REDIS_HOST, port=Config.PORT, log_level="info")

# ─── Router Worker Process ───────────────────────────────────────────────────
def router_worker(max_new_tokens=256):
    model, tokenizer = lazy_load_model(config_)
    model.config.use_cache = True

    device = next(model.parameters()).device
    r = get_redis()
    while True:
        # 等待统一 prompt 队列的请求
        _, raw = r.blpop("prompt")
        data = json.loads(raw)
        session_id = data.get("session_id")
        prompt = data.get("prompt", "")
        answer_key = f"answer:{session_id}"

        # 模型 prefill
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        inputs = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            out = model(input_ids=inputs, use_cache=True, return_dict=True)
            past = out.past_key_values

        eos_id = model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else tokenizer.eos_token_id
        for step in range(max_new_tokens):
            if step == 0:
                last_id = inputs[0, -1].unsqueeze(0).unsqueeze(0)
            else:
                last_id = torch.tensor([[token_id]], device=device)
            with torch.no_grad():
                out = model(input_ids=last_id, past_key_values=past,
                            use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = out.logits
            token_id = logits[0, -1].argmax(-1).item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            r.rpush(answer_key, token_str)
            # 如果生成了结束符，则提前终止
            if token_id == eos_id:
                break

        # 写入结束标志
        r.rpush(answer_key, "[Done]")