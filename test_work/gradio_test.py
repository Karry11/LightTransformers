import gradio as gr
import httpx

def stream_inference(prompt: str):
    """
    向本地 /inference 接口发起流式请求，
    每拿到一段就 yield 出去，让 Gradio 实现“边生成边展示”。
    """
    url = "http://127.0.0.1:8000/inference"
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json={"prompt": prompt}) as resp:
            text = ""
            for chunk in resp.iter_text():
                if chunk:
                    text += chunk
                    yield text

with gr.Blocks() as demo:
    gr.Markdown("## Redis + FastAPI Qwen 推理调试 Demo")
    input_box = gr.Textbox(label="Prompt", placeholder="输入你的提示词…")
    output_box = gr.Textbox(
        label="生成结果",
        placeholder="模型生成会实时显示在这里",
        interactive=False,
        lines=10,
    )
    btn = gr.Button("开始生成")
    # 点击按钮后触发 stream_inference，并开启流式输出
    btn.click(fn=stream_inference, inputs=input_box, outputs=output_box, stream=True)

if __name__ == "__main__":
    demo.launch()
