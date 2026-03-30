import gradio as gr
import requests
import os

# ===================== 你的配置 =====================
API_KEY = os.environ.get('DEEPSEEK_API_KEY')


# ====================================================

def chat(message, history):
    # 构造请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 拼接对话历史
    messages = []
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})

    messages.append({"role": "user", "content": message})

    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(
            url="https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        res = response.json()
        return res["choices"][0]["message"]["content"]

    except Exception as e:
        return f"出错了：{str(e)}"


# 启动界面
demo = gr.ChatInterface(
    fn=chat,
    title="DeepSeek 开源大模型聊天",
    description="Gradio + DeepSeek API 本地运行",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)