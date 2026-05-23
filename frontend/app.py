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
            msg = f"上传成功! {data['uploaded']} 个文件，生成 {data['chunks_indexed']} 个文本块"
            if data.get("skipped"):
                msg += f"\n跳过重复: {', '.join(data['skipped'])}"
            return msg
        return f"上传失败: {data.get('detail', str(data))}"
    except requests.ConnectionError:
        return "无法连接到后端，请先启动: python backend/main.py"


def chat(message, history):
    if not message.strip():
        return ""
    try:
        r = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": message, "top_k": 5, "history": history},
            stream=True, timeout=120,
        )
        answer = ""
        sources = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data.startswith("[DONE]"):
                continue
            if data.startswith("[ERROR]"):
                answer += f"\n错误: {data[7:]}"
                break
            if data.startswith("[SOURCES]"):
                continue
            answer += data
        if answer and sources:
            answer += "\n\n" + sources
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


def list_documents_ui():
    """获取文档列表，返回 Markdown 表格 + 下拉选项"""
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        data = r.json()
        docs = data.get("documents", [])
        if not docs:
            return "暂无已索引的文档", gr.Dropdown(choices=[], interactive=False)
        lines = ["| # | 文件名 | 分块数 |", "|---|---|---|"]
        choices = []
        for i, d in enumerate(docs, 1):
            lines.append(f"| {i} | {d['filename']} | {d['chunks']} |")
            choices.append(d["filename"])
        return "\n".join(lines), gr.Dropdown(choices=choices, interactive=True)
    except requests.ConnectionError:
        return "后端未连接", gr.Dropdown(choices=[], interactive=False)


def delete_document_ui(filename):
    if not filename:
        return "请先选择一个文档"
    try:
        r = requests.delete(f"{BACKEND_URL}/documents/{filename}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return f"已删除: {filename} ({data['chunks_deleted']} 个分块)"
        return f"删除失败: {r.json().get('detail', str(r.text))}"
    except requests.ConnectionError:
        return "后端未连接"


def refresh_ui():
    """刷新状态和文档列表"""
    status = check_health()
    doc_list, dropdown = list_documents_ui()
    return status, doc_list, dropdown


with gr.Blocks(title="RAG 知识库问答") as demo:
    gr.Markdown(
        """
        # RAG 知识库问答系统
        上传文档（PDF/Word/TXT/Markdown），基于文档内容提问，AI 给出带引用的回答。
        """
    )

    with gr.Row():
        status = gr.Textbox(label="系统状态", value="检查中...", interactive=False)
        gr.Button("刷新状态").click(fn=refresh_ui, outputs=[status, doc_list_md, doc_dropdown])

    with gr.Tab("上传文档"):
        file_input = gr.File(
            label="选择文档（支持 PDF / DOCX / TXT / MD）",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt", ".md"],
        )
        upload_btn = gr.Button("上传并索引", variant="primary")
        upload_status = gr.Textbox(label="上传状态", interactive=False)
        upload_btn.click(fn=upload_files, inputs=file_input, outputs=upload_status)

    with gr.Tab("文档管理"):
        with gr.Row():
            with gr.Column(scale=2):
                doc_list_md = gr.Markdown("加载中...")
            with gr.Column(scale=1):
                doc_dropdown = gr.Dropdown(
                    label="选择要删除的文档",
                    choices=[],
                    interactive=False,
                )
                delete_btn = gr.Button("删除选中文档", variant="stop")
                delete_status = gr.Textbox(label="操作结果", interactive=False)
                delete_btn.click(fn=delete_document_ui, inputs=doc_dropdown, outputs=delete_status)

    with gr.Tab("对话问答"):
        gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(height=500, placeholder="上传文档后，在此提问..."),
            textbox=gr.Textbox(placeholder="输入你的问题...", container=False, scale=7),
        )

    demo.load(fn=refresh_ui, outputs=[status, doc_list_md, doc_dropdown])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
