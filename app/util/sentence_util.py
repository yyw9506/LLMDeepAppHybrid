from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
import torch
import faiss
import numpy as np
import os


class SentenceUtil:
    def __init__(self):
        # 加载预训练的BGE模型
        self.model = SentenceTransformer("C:\\Users\wangyy592\PycharmProjects\LLMDeepAppHybrid\\bge-large-zh-v1.5-main")
        self.index = None
        self.lines = None

    def get_embedding(self, line):
        query_embedding = self.model.encode(line, convert_to_tensor=True)
        return query_embedding

    def get_embedding_list(self, line):
        query_embedding = self.model.encode(line, convert_to_tensor=True)
        return query_embedding.tolist()

    def set_index(self, lines):
        # 使用BGE模型将文本转换为向量
        embeddings = self.model.encode(lines)

        # 创建一个FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 使用内积作为相似度度量

        # 将向量添加到索引中
        index.add(embeddings)
        self.lines = lines
        self.index = index

    def get_top_k_similar(self, question, k=3):
        query_embedding = self.model.encode([question])

        # 执行搜索
        D, I = self.index.search(query_embedding, k)

        lista = []
        for i in range(k):
            lista.append(self.lines[I[0][i]])
        return lista
        # for i in range(k):
        #     print(f"相似度: {D[0][i]}, 文本: {self.lines[I[0][i]]}")

    def get_cosine_sim_score(self, embedding1, embedding2):
        return cos_sim(embedding1, embedding2)

def get_knowledge():
    listKnows = []
    with open("C:\\Users\wangyy592\PycharmProjects\LLMDeepAppHybrid\prompts\\repository.txt", 'r', encoding="utf-8") as file:
        strLine = ""
        while True:
            line = file.readline()
            if not line:
                break

            if (line.strip() != ""):
                strLine = strLine + "\n" + line
            else:
                listKnows.append(strLine)
                strLine = ""
        if strLine != "":
            listKnows.append(strLine)
    return listKnows


