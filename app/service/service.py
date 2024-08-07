from app.util.knowledge_util import KnowledgeUtil
from app.util.sentence_util import SentenceUtil
from app.util.model_util import ModelUtil

# 加载本地知识库
app_model = ModelUtil()
sentence_util = SentenceUtil()
knowledge_model = KnowledgeUtil()


def switch_model_domain(domain: str):
    print("切换本地业务知识库")
    KnowledgeUtil.switch_knowledge_module(domain)
    print("切换业务知识库加载完成")


def get_answer_from_model_with_self_consistency(question: str,
                                                vote_n: int,
                                                vote_strategy: str,
                                                threshold: float):
    processed_question = (
        get_compressed_question_from_model_with_self_consistency(
            question,
            vote_n,
            vote_strategy,
            threshold
        )
    )
    refs, indices = knowledge_model.get_top_k_similar(processed_question, 3)
    average_similarity = 0
    filtered_indices = []
    for i in range(0, len(indices)):
        embedding1 = sentence_util.get_embedding(processed_question)
        embedding2 = sentence_util.get_embedding(KnowledgeUtil.lines[indices[i]])
        cosine_similarity_score = sentence_util.get_cosine_sim_score(embedding1, embedding2)
        average_similarity += cosine_similarity_score
        average_similarity /= (i + 1)
        if cosine_similarity_score > 0.7:
            filtered_indices.append(indices[i])

    if len(filtered_indices) > 0:
        return app_model.get_answer_to_question_with_consistency(processed_question, 3, "cosine", 0.6, filtered_indices)
    else:
        print("cannot provide reliable response")
        return "无法提供可靠回答"


def get_compressed_question_from_model_with_self_consistency(question: str,
                                                             vote_n: int,
                                                             vote_strategy: str,
                                                             threshold: float):
    return app_model.get_compressed_question_with_consistency(question, vote_n, vote_strategy, threshold)
