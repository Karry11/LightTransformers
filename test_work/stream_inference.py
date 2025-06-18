from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
from multiprocessing import Process
import uvicorn
import json

# 流式生成模拟器：将 prompt 拆成字符，每0.5s返回一个字符
async def stream_generator(prompt: str):
    for ch in prompt:
        yield ch
        await asyncio.sleep(0.5)
    yield "\n[Done]\n"

# HTTP Server 启动函数（子进程）
def start_http_server():
    app = FastAPI()

    @app.post("/inference")
    async def inference(request: Request):
        body = await request.body()
        data = json.loads(body.decode("utf-8"))
        prompt = data.get("prompt", "")

        print(f"[HTTP Server] Received prompt: {prompt}")
        return StreamingResponse(stream_generator(prompt), media_type="text/plain")

    print("[HTTP Server] Starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 主入口：启动子进程运行 HTTP server
if __name__ == "__main__":
    p = Process(target=start_http_server)
    p.start()
    p.join()  # 如果你希望主进程保持运行，建议换成 while True: ...
