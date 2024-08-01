import faiss
from sentence_transformers import SentenceTransformer


class KnowledgeUtil:
    # 静态数据
    model = SentenceTransformer("C:\\Users\\wangyy592\\PycharmProjects\\LLMDeepAppHybrid\\bge-large-zh-v1.5-main")
    index = None
    lines = []

    @classmethod
    def initialize(cls):
        print("知识库初始化...")
        cls.lines = cls.get_business_knowledge()
        cls.index = cls.to_local_cache_index(cls.lines)
        print("知识库初始化完成")

    @staticmethod
    def get_business_knowledge():
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
        return similar_lines

    @staticmethod
    def switch_knowledge_module(sys: str):
        pass


KnowledgeUtil.initialize()