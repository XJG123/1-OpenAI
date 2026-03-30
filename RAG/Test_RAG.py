from openai import OpenAI
import os

api_key=os.environ['DEEPSEEK_API_KEY']

client=OpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key
)

completion=client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role":"user",
            "content":"你好"
        }
    ]
)

print(completion.choices[0].message.content)