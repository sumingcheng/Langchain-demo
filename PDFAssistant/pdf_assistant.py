from fastapi import FastAPI
from pydantic import BaseModel
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import os
from threading import Lock
from mysql_database import MySQLDatabase
from fastapi.responses import JSONResponse
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
app = FastAPI()
lock = Lock()

# 数据库配置和 MySQLDatabase 类定义
db = MySQLDatabase("localhost", "root", "root", "text_data")

# 模型和 Tokenizer 的加载
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# FAISS 索引初始化
d = 768  # 嵌入向量的维度
index_path = "faiss_index_pdf.bin"


def load_faiss_index(index_path="faiss_index_pdf.bin"):
    with lock:
        if os.path.exists(index_path):
            print("Loading existing FAISS index.")
            return faiss.read_index(index_path)
        else:
            print("Creating new FAISS index.")
            return faiss.IndexFlatL2(d)


index = load_faiss_index(index_path)


def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texts = [page.get_text() for page in doc]
        doc.close()
        return texts
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []


def get_text_embedding(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    except Exception as e:
        print(f"Error getting text embedding: {e}")
        return None


def process_pdf_and_update_index(pdf_path):
    with lock:
        if not os.path.exists(index_path):
            texts = extract_text_from_pdf(pdf_path)
            for text in texts:
                vector = get_text_embedding(text)
                if vector is not None:
                    index.add(vector)
                    vector_id = index.ntotal - 1
                    db.save_text(vector_id, text)
            faiss.write_index(index, index_path)
            print("已处理 PDF 并更新了 FAISS 索引。")
        else:
            print("FAISS 索引已存在。跳过 PDF 处理。")


@app.on_event("startup")
def startup_event():
    pdf_path = "test.pdf"  # Update with your PDF path
    process_pdf_and_update_index(pdf_path)


class QueryItem(BaseModel):
    query: str


@app.get("/search/", response_class=JSONResponse)
async def search_embedding(query: str):
    query_embedding = get_text_embedding(query)
    with lock:
        D, I = index.search(query_embedding, k=1)
        matched_index = I[0][0]  # I[0][0] 可能是 numpy.int64 类型
        matched_text = db.get_text_by_id(int(matched_index))  # 转换为 Python int

    # D[0][0] 可能是 numpy.float64 类型，需要转换为 Python float
    optimized_response = {
        "matched_index": int(matched_index),  # 确保是 Python 基本数据类型
        "distance": float(D[0][0]),  # 确保是 Python 基本数据类型
        "matched_summary": "。".join(matched_text.split("。")[:2]) + "。",
        "matched_text": matched_text.replace(" \n", "\n\n")
    }
    return optimized_response
