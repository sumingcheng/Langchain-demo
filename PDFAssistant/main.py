from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os


class TextItem(BaseModel):
    text: str


app = FastAPI()

# 加载模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# 嵌入向量的维度
d = 768


def load_faiss_index(index_path="faiss_index.bin"):
    if os.path.exists(index_path):
        print("Loading FAISS index from file.")
        return faiss.read_index(index_path)
    else:
        print("FAISS index file not found, creating a new index.")
        return faiss.IndexFlatL2(d)


# 启动时加载或创建索引
index = load_faiss_index()


def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


@app.post("/embed/")
async def create_embedding(item: TextItem):
    print(f"Creating embedding for: {item.text}")
    embedding = get_text_embedding(item.text)
    index.add(embedding)
    faiss.write_index(index, "faiss_index.bin")
    return {"message": "嵌入已创建并添加到 FAISS 索引，索引已保存。"}


@app.get("/search/")
async def search_embedding(query: str):
    query_embedding = get_text_embedding(query)
    D, I = index.search(query_embedding, k=1)
    return {"index": I.tolist(), "distance": D.tolist()}
