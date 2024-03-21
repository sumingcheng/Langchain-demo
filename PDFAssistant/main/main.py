# main.py
from fastapi import FastAPI
from pdf_processor import PDFProcessor
from text_embedder import TextEmbedder
from faiss_index import FaissIndex
from mysql_database import MySQLDatabase
import jieba

app = FastAPI()
embedder = TextEmbedder()
faiss_index = FaissIndex()
pdf_processor = PDFProcessor()
db = MySQLDatabase("localhost", "root", "root", "text_data")


@app.on_event("startup")
async def startup_event():
    global faiss_index
    pdf_path = "test.pdf"  # 根据实际PDF文件路径修改
    texts = pdf_processor.extract_text_from_pdf(pdf_path)
    for text in texts:
        # 使用jieba分词对文本进行分句
        sentences = [sentence for sentence in jieba.cut(text, cut_all=False) if sentence.strip()]
        for sentence in sentences:
            vector = embedder.get_text_embedding(sentence)
            # 由于每个句子都进行向量化，因此每个向量都应该添加到索引中
            faiss_index.add_to_index(vector)
            # 注意：以下两行需要在循环中执行，以确保每个句子都被保存
            vector_id = faiss_index.index.ntotal - 1
            db.save_text(vector_id, sentence)


@app.get("/search/")
async def search_embedding(query: str):
    query_embedding = embedder.get_text_embedding(query)
    D, I = faiss_index.index.search(query_embedding, k=1)
    matched_index = I[0][0]
    matched_text = db.get_text_by_id(matched_index)

    if matched_text is None:
        return {"error": "No match found"}

    return {
        "matched_index": int(matched_index),
        "distance": float(D[0][0]),
        "matched_text": matched_text
    }
