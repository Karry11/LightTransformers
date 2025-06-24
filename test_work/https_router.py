import multiprocessing as mp
from multiprocessing import Process
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import json
import time
input_path = "/mnt/e/LocalModelList/Qwen2.5-0.5B/"
# input_path = "/mnt/s/NLP/LocalModel/qwen2-0.5b/"
# ─── HTTP Server Process ─────────────────────────────────────────────────────

def start_http_server(request_queue, response_queue):
    # 仅加载 tokenizer，无需初始化模型
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        input_path,
        trust_remote_code=True,
        local_files_only=True
    )

    app = FastAPI()

    @app.post("/inference")
    async def inference(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        # 1) Tokenize prompt
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # 2) 写入请求队列
        request_queue.put(ids)
        # 3) 返回 StreamingResponse，从 response_queue 读取结果
        def event_generator():
            while True:
                token = response_queue.get()
                yield token
                if token == "\n[Done]\n":
                    break
        return StreamingResponse(event_generator(), media_type="text/plain")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# ─── Router Process ───────────────────────────────────────────────────────────

def router_worker(request_queue, response_queue, max_new_tokens=256):
    # 在此进程里加载模型和 tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    MODEL_PATH = input_path
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model.config.use_cache = True
    device = next(model.parameters()).device

    while True:
        # 阻塞，等待新的 prompt id 列表
        input_ids_list = request_queue.get()
        # 转成 tensor 并 prefill
        inputs = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=inputs, use_cache=True, return_dict=True)
            past = out.past_key_values
        # 逐 token 生成并写回 response_queue
        for _ in range(max_new_tokens):
            # 取最后一个生成 id
            if _ == 0:
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
            response_queue.put(token_str)
        # 生成结束标记
        response_queue.put("\n[Done]\n")

# ─── 主入口：创建队列和进程 ─────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    req_queue = manager.Queue()
    res_queue = manager.Queue()

    p_http = Process(target=start_http_server, args=(req_queue, res_queue))
    p_router = Process(target=router_worker, args=(req_queue, res_queue))
    p_http.daemon = True
    p_router.daemon = True
    p_http.start()
    p_router.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        p_http.terminate()
        p_router.terminate()
        p_http.join()
        p_router.join()
