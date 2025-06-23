# import multiprocessing as mp
# from multiprocessing import Process
# from fastapi import FastAPI, Request
# from fastapi.responses import StreamingResponse
# import uvicorn
# import redis
# import json
# import time
#
# # ─── Redis 连接池 ───────────────────────────────────────────────────────────
# redis_pool = redis.ConnectionPool(
#     host="127.0.0.1",
#     port=6379,
#     db=0,
#     max_connections=20,
# )
#
# def get_redis():
#     return redis.Redis(connection_pool=redis_pool)
#
# # ─── HTTP Server Process ─────────────────────────────────────────────────────
# def start_http_server():
#     app = FastAPI()
#
#     @app.post("/inference")
#     async def inference(request: Request):
#         data = await request.json()
#         prompt = data.get("prompt", "")
#         if not prompt:
#             return {"error": "no prompt provided"}
#
#         r = get_redis()
#         # 1) 将 prompt 推入 Redis 列表
#         r.rpush("prompt", prompt)
#
#         # 2) 返回流式响应，从 answer 列表读取生成结果
#         def event_generator():
#             while True:
#                 # BLPOP 会阻塞直到有元素可读
#                 _, raw = r.blpop("answer")
#                 token = raw.decode("utf-8")
#                 yield token
#                 if token == "[Done]":
#                     break
#
#         return StreamingResponse(event_generator(), media_type="text/plain")
#
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
#
#
# # ─── Router Worker Process ───────────────────────────────────────────────────
# def router_worker(max_new_tokens=256):
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     import torch
#
#     MODEL_PATH = "/mnt/s/NLP/LocalModel/qwen2-0.5b/"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     model.config.use_cache = True
#     device = next(model.parameters()).device
#
#     r = get_redis()
#     while True:
#         # 1) 等待新的 prompt
#         _, raw = r.blpop("prompt")
#         prompt = raw.decode("utf-8")
#
#         # 2) Prefill
#         input_ids = tokenizer.encode(prompt, add_special_tokens=False)
#         inputs = torch.tensor([input_ids], device=device)
#         with torch.no_grad():
#             out = model(input_ids=inputs, use_cache=True, return_dict=True)
#             past = out.past_key_values
#
#         # 3) 逐 token 生成并推入 answer 列表
#         for step in range(max_new_tokens):
#             if step == 0:
#                 last_id = inputs[0, -1].unsqueeze(0).unsqueeze(0)
#             else:
#                 last_id = torch.tensor([[token_id]], device=device)
#             with torch.no_grad():
#                 out = model(input_ids=last_id, past_key_values=past,
#                             use_cache=True, return_dict=True)
#                 past = out.past_key_values
#                 logits = out.logits
#             token_id = logits[0, -1].argmax(-1).item()
#             token_str = tokenizer.decode([token_id], skip_special_tokens=True)
#             r.rpush("answer", token_str)
#
#         # 4) 写入结束标志
#         r.rpush("answer", "[Done]")
#
# # ─── 主入口：启动两个进程 ────────────────────────────────────────────────────
# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#
#     p_http = Process(target=start_http_server)
#     p_router = Process(target=router_worker)
#
#     p_http.start()
#     p_router.start()
#
#     try:
#         p_http.join()
#         p_router.join()
#     except KeyboardInterrupt:
#         p_http.terminate()
#         p_router.terminate()
#         p_http.join()
#         p_router.join()
import multiprocessing as mp
from multiprocessing import Process
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import redis
import json
import time

# ─── Redis 连接池 ───────────────────────────────────────────────────────────
redis_pool = redis.ConnectionPool(
    host="127.0.0.1",
    port=6379,
    db=0,
    max_connections=20,
)

def get_redis():
    return redis.Redis(connection_pool=redis_pool)

# ─── HTTP Server Process ─────────────────────────────────────────────────────
def start_http_server():
    app = FastAPI()

    @app.post("/inference")
    async def inference(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            return {"error": "no prompt provided"}

        r = get_redis()
        # 1) 将 prompt 推入 Redis 列表
        r.rpush("prompt", prompt)

        # 2) 返回流式响应，从 answer 列表读取生成结果
        def event_generator():
            while True:
                # BLPOP 会阻塞直到有元素可读
                _, raw = r.blpop("answer")
                token = raw.decode("utf-8")
                yield token
                if token == "[Done]":
                    # 删除 answer key
                    r.delete("answer")
                    break

        return StreamingResponse(event_generator(), media_type="text/plain")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# ─── Router Worker Process ───────────────────────────────────────────────────
def router_worker(max_new_tokens=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    MODEL_PATH = "/mnt/s/NLP/LocalModel/qwen2-0.5b/"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = True
    device = next(model.parameters()).device

    r = get_redis()
    while True:
        # 1) 等待新的 prompt
        _, raw = r.blpop("prompt")
        prompt = raw.decode("utf-8")

        # 2) Prefill
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        inputs = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            out = model(input_ids=inputs, use_cache=True, return_dict=True)
            past = out.past_key_values

        # 3) 逐 token 生成并推入 answer 列表
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
            r.rpush("answer", token_str)

        # 4) 写入结束标志
        r.rpush("answer", "[Done]")

# ─── 主入口：启动两个进程 ────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    p_http = Process(target=start_http_server)
    p_router = Process(target=router_worker)

    p_http.start()
    p_router.start()

    try:
        p_http.join()
        p_router.join()
    except KeyboardInterrupt:
        p_http.terminate()
        p_router.terminate()
        p_http.join()
        p_router.join()
