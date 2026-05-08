"""
Gradio 前端 — RAG 知识库问答系统可视化界面
启动: python frontend/app.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
footer { display: none !important; }
"""


def upload_files(files):
    if not files:
        return "请选择文件"
    file_objs = []
    for fp in files:
        file_objs.append(("files", (os.path.basename(fp), open(fp, "rb"))))
    try:
        r = requests.post(f"{BACKEND_URL}/upload", files=file_objs)
        data = r.json()
        if r.status_code == 200:
            return f"上传成功! {data['uploaded']} 个文件，生成 {data['chunks_indexed']} 个文本块"
        return f"上传失败: {data.get('detail', str(data))}"
    except requests.ConnectionError:
        return "无法连接到后端，请先启动: python backend/main.py"


def chat(message, history):
    if not message.strip():
        return ""
    try:
        r = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": message, "top_k": 5},
            stream=True, timeout=60,
        )
        answer = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data.startswith("[SOURCES]") or data.startswith("[DONE]"):
                continue
            if data.startswith("[ERROR]"):
                answer += f"\n错误: {data[7:]}"
                break
            answer += data
        return answer
    except requests.ConnectionError:
        return "无法连接到后端，请先确保 API 服务已启动"


def check_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        data = r.json()
        return f"服务正常 | 已索引 {data.get('indexed_chunks', 0)} 个文本块"
    except Exception:
        return "后端未连接"


with gr.Blocks(title="RAG 知识库问答") as demo:
    gr.Markdown(
        """
        # RAG 知识库问答系统
        上传文档（PDF/Word/TXT/Markdown），基于文档内容提问，AI 给出带引用的回答。
        """
    )

    with gr.Row():
        status = gr.Textbox(label="系统状态", value="检查中...", interactive=False)
        gr.Button("刷新状态").click(fn=check_health, outputs=status)

    with gr.Tab("上传文档"):
        file_input = gr.File(
            label="选择文档（支持 PDF / DOCX / TXT / MD）",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt", ".md"],
        )
        upload_btn = gr.Button("上传并索引", variant="primary")
        upload_status = gr.Textbox(label="上传状态", interactive=False)
        upload_btn.click(fn=upload_files, inputs=file_input, outputs=upload_status)

    with gr.Tab("对话问答"):
        gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(height=500, placeholder="上传文档后，在此提问..."),
            textbox=gr.Textbox(placeholder="输入你的问题...", container=False, scale=7),
        )

    demo.load(fn=check_health, outputs=status)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
