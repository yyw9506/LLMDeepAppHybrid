from app.util.model_util import ModelUtil
from app.util.sentence_util import SentenceUtil, get_knowledge

# 加载本地知识库
print("加载本地知识库...")
bgd_model = SentenceUtil()
lines = get_knowledge()
bgd_model.set_index(lines)
print("完成本地知识库加载")
app_model = ModelUtil()


def get_answer_from_model(message: str):
    refs = bgd_model.get_top_k_similar(message, 3)
    return app_model.get_response(message, refs)