# 导入操作系统相关功能模块
import os
# 导入用于便捷获取字典/对象中指定项的工具
from operator import itemgetter
# 导入面向对象的文件路径处理工具，替代传统os.path
from pathlib import Path

# 导入LangChain核心：提示词模板，用于构建给大模型的输入指令
from langchain_core.prompts import PromptTemplate
# 导入输出解析器，将大模型返回结果解析为字符串格式
from langchain_core.output_parsers import StrOutputParser
# 导入LangChain链式调用核心组件：透传参数、并行执行任务
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# 导入文档加载器：加载指定目录下的所有文档、加载纯文本文件
from langchain_community.document_loaders import DirectoryLoader, TextLoader,PyPDFLoader
# 导入OpenAI大模型调用封装类
from langchain_openai import ChatOpenAI
# 导入Chroma向量数据库，用于存储文本向量并做相似度检索
from langchain_chroma import Chroma
# 导入递归字符文本分割器，用于将长文本切分成小块
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入HuggingFace开源词向量模型封装类
from langchain_huggingface import HuggingFaceEmbeddings

#1、设置模型
llm=ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com",api_key=os.environ.get('DEEPSEEK_API_KEY'))
embedding_model = HuggingFaceEmbeddings(model_name=r"G:\000AIWorkeSpace\AgentWorkSpace\my_ai\model_dir\BAAI\bge-large-zh-v1___5")
#2、设置数据处理--加载、分块、存储、检索
##加载
    # 定义知识库文件存放的目录路径（文件夹名称：my_knowledge）
    # 使用 Path 类进行面向对象的路径处理，比 os.path 更简洁、跨平台
file_dir=Path('my_knowledge')
##分块  每500字符分一块，滑动窗口分块chunk_overlap=100
    # 【核心：文本分块器】将长文本递归切分成固定大小的小片段，用于向量检索
    # chunk_size：每个文本块的最大字符长度（500字符≈300~400个中文汉字）
    # chunk_overlap：块与块之间的重叠字符数（100），防止语义被切断
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
##存储 向量存储
    # 【初始化向量数据库】
    # 创建/加载 Chroma 向量库实例
    # embedding_function：指定向量生成模型（把文本转成数值向量）
    # persist_directory：向量库持久化存储路径（关闭程序后数据不丢失）
vector_stor=Chroma(embedding_function=embedding_model,persist_directory="./chroma_v3")
##检索
    # 【构建检索器】从向量库中获取与用户问题最相关的文档片段
    # search_kwargs={"k": 5}：表示每次检索返回 最相似的前5条 文本块
retriever=vector_stor.as_retriever(search_kwargs={"k":5})
##提示词模板
    # 【RAG提示词模板】定义大模型的回答规则和格式
    # 核心作用：约束模型必须基于提供的上下文回答，禁止编造信息
prompt_template = PromptTemplate.from_template("""你是一个严谨的RAG助手。
请根据以下提供的上下文信息来回答问题。
如果上下文信息不足以回答问题，请直接说“根据提供的信息无法回答”。
如果回答时使用了上下文中的信息，在回答后输出使用了哪些上下文，
上下文信息：
{context}
------------
问题：{question}""")

#3、编排“链”，串起来
chain = {"question":RunnablePassthrough()} | RunnablePassthrough.assign(context = itemgetter("question") | retriever) | prompt_template | llm | StrOutputParser()



if __name__ == '__main__':
    #chain.invoke("一天上多久班")
    #4、初始化数据库--文档的加载和拆分  只在数据库文件更新时进行调用
    #docs = DirectoryLoader(str(file_dir),loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"}).load()#加载文档
    #docs = text_splitter.split_documents(docs)#切分文档
    #vector_stor.add_documents(docs)#存储文档
    print(chain.invoke("一天上多久班"))