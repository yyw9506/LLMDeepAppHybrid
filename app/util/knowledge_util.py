import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class KnowledgeUtil:

    def __init__(self):
        pass

    # 静态数据
    model = SentenceTransformer("C:\\Users\\wangyy592\\PycharmProjects\\LLMDeepAppHybrid\\bge-large-zh-v1.5-main")
    index = None
    lines = []

    @classmethod
    def initialize(cls):
        print("知识库初始化...")
        cls.lines = cls.get_business_knowledge()
        print("正在创建索引...")
        cls.index = cls.to_local_cache_index(cls.lines)
        print("知识库初始化完成")

    @staticmethod
    def get_default_knowledge():
        list_knows = []
        with open("C:\\Users\\wangyy592\\PycharmProjects\\LLMDeepAppHybrid\\prompts\\repository.txt", 'r',
                  encoding="utf-8") as file:
            str_line = ""
            while True:
                line = file.readline()
                if not line:
                    break
                if line.strip() != "":
                    str_line = str_line + "\n" + line
                else:
                    list_knows.append(str_line)
                    str_line = ""
            if str_line != "":
                list_knows.append(str_line)
        return list_knows

    @staticmethod
    def get_business_knowledge():
        df = pd.read_csv("C:\\Users\wangyy592\PycharmProjects\LLMDeepApp\prompts\\business_chat.csv")
        list_knows = []
        for i, row in df.iterrows():
            # drop low value record
            if len(row["background"]) <= 24 or len(str(row["advice"])) <= 10 or len(str(row["last_reply"])) <= 10:
                continue
            if len(list_knows) < 400:
                list_knows.append(row["question"])
            else:
                break
        return list_knows

    @staticmethod
    def to_local_cache_index(lines):
        # convert lines to vector
        vectors = KnowledgeUtil.model.encode(lines)
        # create FAISS index
        dimension = vectors.shape[1]
        # inner product
        index = faiss.IndexFlatIP(dimension)
        # add vector to index
        index.add(vectors)
        return index

    @staticmethod
    def get_top_k_similar(question: str, k: int):
        query_vector = KnowledgeUtil.model.encode([question])
        # search top k similar
        distances, indices = KnowledgeUtil.index.search(query_vector, k)
        similar_lines = [KnowledgeUtil.lines[i] for i in indices[0]]
        return similar_lines, indices[0]

    @staticmethod
    def switch_knowledge_module(sys: str):
        if sys == "business":
            KnowledgeUtil.get_business_knowledge()
        else:
            pass


KnowledgeUtil.initialize()
