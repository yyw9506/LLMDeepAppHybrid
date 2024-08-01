# encoding:utf-8
from app.util.knowledge_util import KnowledgeUtil
from app.util.model_util import ModelUtil

# 加载本地知识库
app_model = ModelUtil()
knowledge_model = KnowledgeUtil()


def switch_model_domain(domain: str):
    if domain == "business":
        print("加载本地业务知识库")
        KnowledgeUtil.switch_knowledge_module(domain)
        print("本地业务知识库加载完成")
    elif domain == "code":
        print("加载本地代码知识库")
        # to-do

        # to-do
        print("本地代码知识库加载完成")
    else:
        pass


def get_answer_from_model(message: str):
    refs = knowledge_model.get_top_k_similar(message, 3)
    return app_model.get_user_question_response(message, refs)


def get_answer_from_model_with_self_consistency(question: str,
                                                vote_n: int,
                                                vote_strategy: str,
                                                threshold: float):
    processed_question = (
        get_compressed_question_from_model_with_self_consistency(
            question,
            vote_n,
            vote_strategy,
            threshold)
    )
    refs = knowledge_model.get_top_k_similar(processed_question, 3)
    print(refs)
    # to-do
    for i in range(0, vote_n):
        pass

    # to-do
    return ""


def get_compressed_question_from_model_with_self_consistency(question: str,
                                                             vote_n: int,
                                                             vote_strategy: str,
                                                             threshold: float):

    return app_model.get_compressed_user_question_with_consistency(question, vote_n, vote_strategy, threshold)
