from fastapi import FastAPI, Request
import uvicorn


def run_http_server():
    app = FastAPI()

    @app.post("/inference")
    async def inference(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        print(f"[HTTP Server] Received prompt: {prompt}")
        return {"message": "Prompt received", "prompt": prompt}

    print("[HTTP Server] Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)