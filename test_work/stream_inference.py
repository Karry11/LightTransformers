# from fastapi import FastAPI, Request
# from fastapi.responses import StreamingResponse
# import asyncio
# from multiprocessing import Process
# import uvicorn
# import json
#
# # 流式生成模拟器：将 prompt 拆成字符，每0.5s返回一个字符
# async def stream_generator(prompt: str):
#     for ch in prompt:
#         yield ch
#         await asyncio.sleep(0.5)
#     yield "\n[Done]\n"
#
# # HTTP Server 启动函数（子进程）
# def start_http_server():
#     app = FastAPI()
#
#     @app.post("/inference")
#     async def inference(request: Request):
#         body = await request.body()
#         data = json.loads(body.decode("utf-8"))
#         prompt = data.get("prompt", "")
#
#         print(f"[HTTP Server] Received prompt: {prompt}")
#         return StreamingResponse(stream_generator(prompt), media_type="text/plain")
#
#     print("[HTTP Server] Starting on http://0.0.0.0:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
#
# # 主入口：启动子进程运行 HTTP server
# if __name__ == "__main__":
#     p = Process(target=start_http_server)
#     p.start()
#     p.join()  # 如果你希望主进程保持运行，建议换成 while True: ...
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
from multiprocessing import Process
import multiprocessing as mp
import uvicorn
import json
import time

# 载入 Qwen2 模型与 tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/mnt/s/NLP/LocalModel/qwen2-0.5b/"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = True

device = next(model.parameters()).device

def manual_stream(prompt: str, max_new_tokens: int = 512):
    """
    同步生成器，以逐 token 形式输出模型推理结果字符串
    """
    # 编码并 prefill
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(
            **inputs,
            use_cache=True,
            return_dict=True
        )
        past = out.past_key_values

    # 先输出原始 prompt
    # yield prompt

    # 逐步生成
    for _ in range(max_new_tokens):
        # 准备输入 id
        if _ == 0:
            last_id = inputs["input_ids"][0, -1].unsqueeze(0).unsqueeze(0)
        else:
            last_id = torch.tensor([[token_id]], device=device)

        with torch.no_grad():
            out = model(
                input_ids=last_id,
                past_key_values=past,
                use_cache=True,
                return_dict=True
            )
            past = out.past_key_values
            logits = out.logits

        # 贪心取 token
        token_id = logits[0, -1].argmax(-1).item()
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        yield token_str

    yield "\n[Done]\n"

async def stream_generator(prompt: str):
    # 将同步生成器封装为异步
    for chunk in manual_stream(prompt):
        yield chunk
        await asyncio.sleep(0)  # 让出事件循环

# HTTP Server 启动函数（子进程）

def start_http_server():
    app = FastAPI()

    @app.post("/inference")
    async def inference(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        print(f"[HTTP Server] Received prompt: {prompt}")
        return StreamingResponse(
            stream_generator(prompt),
            media_type="text/plain"
        )

    print("[HTTP Server] Starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 主入口：启动子进程并保持主进程运行

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = Process(target=start_http_server)
    p.daemon = True
    p.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down HTTP server...")
        p.terminate()
        p.join()
