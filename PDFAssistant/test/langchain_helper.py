import asyncio
import os
import pickle
from dotenv import load_dotenv
import fitz  # PyMuPDF, 用于处理PDF文件
from langchain_community.vectorstores import FAISS  # 用于向量相似性搜索
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型

# 加载环境变量
load_dotenv()

# 全局变量
EMBEDDINGS_CACHE_DIRECTORY = "./embeddings_cache"

# 确保嵌入向量缓存目录存在
if not os.path.exists(EMBEDDINGS_CACHE_DIRECTORY):
    os.makedirs(EMBEDDINGS_CACHE_DIRECTORY)


async def setup_openai_embeddings():
    """异步初始化OpenAI嵌入模型。"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY环境变量未设置。")
    return OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")


async def extract_pdf_text(pdf_path):
    """从PDF文档中异步提取文本。"""
    doc = await asyncio.to_thread(fitz.open, pdf_path)  # 在线程中运行以避免阻塞
    tasks = [asyncio.to_thread(page.get_text) for page in doc]  # 每页调用get_text在线程中运行
    texts = await asyncio.gather(*tasks)
    return texts


async def cache_embeddings(pdf_path, embeddings_file):
    """异步生成文本的嵌入向量并进行缓存。"""
    extracted_texts = await extract_pdf_text(pdf_path)
    openai_embeddings = await setup_openai_embeddings()
    embeddings = await openai_embeddings.embed_texts(extracted_texts)
    with open(embeddings_file, "wb") as ef:
        pickle.dump(embeddings, ef)
    return embeddings


def check_embeddings_cache(filename):
    """检查嵌入向量缓存。"""
    embeddings_file = os.path.join(EMBEDDINGS_CACHE_DIRECTORY, f"{filename}.embeddings")
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "rb") as ef:
            return pickle.load(ef)
    return None


async def query_pdf_content(query_text, pdf_texts_or_embeddings):
    """根据文本查询PDF内容。"""
    openai_embeddings = await setup_openai_embeddings()
    if isinstance(pdf_texts_or_embeddings, list):
        embeddings = await openai_embeddings.embed_texts(pdf_texts_or_embeddings)
    else:
        embeddings = pdf_texts_or_embeddings
    query_embedding = await openai_embeddings.embed_query(query_text)
    faiss_store = FAISS.from_embeddings(embeddings)
    results = faiss_store.similarity_search(query_embedding, k=1)
    return [result['text'] for result in results]


async def main(pdf_path, query_text):
    """主函数处理PDF和查询内容。"""
    # 检查是否已缓存嵌入向量
    embeddings = check_embeddings_cache(os.path.basename(pdf_path))
    if embeddings is None:
        # 如果没有缓存，则提取文本并缓存嵌入向量
        embeddings_file = os.path.join(EMBEDDINGS_CACHE_DIRECTORY, f"{os.path.basename(pdf_path)}.embeddings")
        embeddings = await cache_embeddings(pdf_path, embeddings_file)

    # 查询PDF内容
    results = await query_pdf_content(query_text, embeddings)
    for result in results:
        print(result)


if __name__ == "__main__":
    pdf_path = "../main/test.pdf"
    query_text = "总结一下这个PDF，说了什么？"
    asyncio.run(main(pdf_path, query_text))
