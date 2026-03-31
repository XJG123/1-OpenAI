from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

model=SentenceTransformer(r"G:\000AIWorkeSpace\AgentWorkSpace\my_ai\model_dir\BAAI\bge-large-zh-v1___5")
#client=chromadb.Client()  #写法一：使用内存数据库
client=chromadb.PersistentClient("./chroma_v2")#方式二：文件数据库
#创建集合
collection=client.get_or_create_collection(
    name="jian_guo",
    metadata={
        "描述":"这是文本文件的向量数据库",
        "创建时间":str(datetime.now()),
        "hnsw:space":"cosine"#用余弦计算相似度--特点：范围【0，1】
    }
)

def txt_2db():
    # 1、加载所有文件
    path_list = list(Path("my_knowledge").glob("*.txt"))#？？？只支持txt，需要支持pdf、word、ppt等
    text_list = []  # 文本内容
    # 1.找到和问题相关的文件
    for path in path_list:  # A.复杂度？
        text = path.read_text(encoding='utf-8')
        text_list.append(text)

    # 2、进行向量嵌入
    embeddings = model.encode(text_list)
    # 3、存入数据库
    collection.add(
        embeddings = embeddings.tolist(), #所有的向量放进去，向量只有自己能认识，无法给其他大模型去解析
        documents = text_list,            #文本存进去
        metadatas=[{"id":i}for i in text_list],    #元数据
        ids=[f"doc_{i}" for i,_ in enumerate(text_list)]  #ID
    )

    print(f'数据库中的数据量:{collection.count()}')

if __name__ == '__main__':
    txt_2db()
    #query=["能飞多久"]
    # query = ["考勤"]
    #
    # query_emedding=model.encode(query)
    # data=collection.query(query_emedding.tolist(),n_results=5)
    # text_list=data['documents'][0]#存在向量数据库里的内容
    # #print(text_list)
    # for t in text_list:
    #     print(t)