from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import jieba


class TextEmbedder:
    def __init__(self, model_name="bert-base-chinese", chunk_size=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.chunk_size = chunk_size

    def get_text_embedding(self, text: str):
        # 使用jieba进行中文分句
        sentences = [sentence for sentence in jieba.cut(text, cut_all=False) if sentence.strip()]
        chunk_embeddings = []

        for sentence in sentences:
            # 对每个句子进行处理，确保不超过模型输入长度
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=self.chunk_size)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 计算每个句子的嵌入向量
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            chunk_embeddings.append(chunk_embedding)

        # 如果有计算出的嵌入向量，计算所有句子嵌入向量的平均值；否则返回一个全零向量
        if chunk_embeddings:
            all_embeddings = np.mean(np.vstack(chunk_embeddings), axis=0, keepdims=True)
        else:
            all_embeddings = np.zeros((1, 768))  # 保持嵌入向量维度的一致性
        return all_embeddings
