from modelscope import snapshot_download

model_dir=snapshot_download(
    model_id='BAAI/bge-large-zh-v1.5',
    cache_dir=r'G:\000AIWorkeSpace\AgentWorkSpace\my_ai\model_dir'
)

#下载完成后本地模型路径：G:\000AIWorkeSpace\AgentWorkSpace\my_ai\model_dir\BAAI\bge-large-zh-v1___5