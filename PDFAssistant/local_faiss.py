from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os
from threading import Lock
from mysql_database import MySQLDatabase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TextItem(BaseModel):
    text: str


app = FastAPI()
db = MySQLDatabase("localhost", "root", "root", "text_data")

# 模型和Tokenizer的加载
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# 嵌入向量的维度
d = 768

# 初始化一个锁
lock = Lock()


def load_faiss_index(index_path="faiss_index.bin"):
    """安全地加载或创建FAISS索引。"""
    if os.path.exists(index_path):
        print("从文件加载FAISS索引。")
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"FAISS索引加载失败: {e}")
    else:
        print("FAISS索引文件未找到，创建新索引。")
        return faiss.IndexFlatL2(d)


# 启动时加载或创建索引
index = load_faiss_index()


def get_text_embedding(text: str):
    """获取文本的嵌入表示。"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


@app.post("/embed/")
async def create_embedding(item: TextItem):
    """创建嵌入并添加到FAISS索引中，并将文本及其索引保存到数据库中。"""
    print(f"为以下内容创建嵌入: {item.text}")
    embedding = get_text_embedding(item.text)
    with lock:  # 使用锁来确保线程安全
        index.add(embedding)
        # 注意：在添加后，FAISS的ntotal属性会增加，所以我们使用当前ntotal减1作为新嵌入向量的索引
        vector_id = index.ntotal - 1
        db.save_text(vector_id, item.text)  # 将文本及其索引保存到数据库
        faiss.write_index(index, "faiss_index.bin")
    return {"message": "索引已保存，文本已保存到数据库。", "vector_id": vector_id}


@app.get("/search/")
async def search_embedding(query: str):
    """搜索嵌入，并根据找到的索引从数据库中检索文本。"""
    query_embedding = get_text_embedding(query)
    with lock:  # 使用锁来确保线程安全
        D, I = index.search(query_embedding, k=2)  # 假设我们只查找最匹配的一个结果
        matched_index = I[0][0]  # 最匹配的索引
        matched_text = db.get_text_by_id(matched_index)  # 使用数据库类根据索引检索文本
    return {"index": I.tolist(), "distance": D.tolist(), "matched_text": matched_text}
