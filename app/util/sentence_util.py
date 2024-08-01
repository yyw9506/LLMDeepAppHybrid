from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
import faiss


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

    def get_cosine_sim_score(self, embedding1, embedding2):
        return cos_sim(embedding1, embedding2)
