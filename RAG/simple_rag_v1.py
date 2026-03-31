from openai import OpenAI
import os
from pathlib import Path

api_key=os.environ['DEEPSEEK_API_KEY']

client=OpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key
)

#步骤一：实现一个简单的“关键词匹配”检索器
def retrieval(query):
    context=""
    # 0.遍历所有文件
    path_list=list(Path("my_knowledge").glob("*.txt"))
    print(path_list)
    # 1.找到和问题相关的文件
    for path in path_list:          #A.复杂度？
        if path.stem in query:      #B.检索机制？   C.【向量？】   D.资料长度？
            context += path.read_text(encoding="utf-8")
            context += "\n\n"


    # 2.相关文件的内容提取出来
    # 3.添加到context中

    return context

#print(retrieval("requirements是什么"))

#步骤二：增强Query
def augmented(query,context=""):
    if not context:
        return f"请简要回答下面问题：{query}"
    else:
        prompt=f"""请根据上下文的信息来回答问题，如果上下文信息不足以回答问题，请直接说：“根据提供的上下文信息，无法回答”
    上下文：
    {context}
    
    问题：{query}"""

        #检查长度是否超过上下文上限

        return prompt

#print(augmented("你是谁","大飞是一个AI助手"))
#步骤三：生成回答

def generation(prompt):
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    query="简要介绍一下探索者x100"
    # print("==不使用RAG===")
    # print(generation(query))

    print("==使用RAG==")
    context=retrieval(query)
    prompt=augmented(query,context)
    resp=generation(prompt)
    print(resp)

