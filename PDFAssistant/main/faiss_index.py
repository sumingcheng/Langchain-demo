# faiss_index.py
import faiss
import os


class FaissIndex:
    def __init__(self, index_path="./faiss_index_pdf.bin", dimension=768):
        self.index_path = index_path
        self.dimension = dimension
        self.index = self.load_faiss_index()

    def load_faiss_index(self):
        if os.path.exists(self.index_path):
            print("Loading existing FAISS index.")
            return faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS index.")
            return faiss.IndexFlatL2(self.dimension)

    def add_to_index(self, vector):
        """向FAISS索引中添加向量"""
        if vector is not None and vector.size > 0:
            self.index.add(vector)
            faiss.write_index(self.index, self.index_path)
