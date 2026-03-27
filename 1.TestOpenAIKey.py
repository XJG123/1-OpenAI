import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('APPLE_DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是个渣男，回复不超过30字"},
        {"role": "user", "content": "我讨厌你"},
    ],
    temperature=2,
    stream=False,
)

print(response.choices[0].message.content)

