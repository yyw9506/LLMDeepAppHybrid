# encoding:utf-8
import requests
import json
import numpy as np
import pandas as pd
from app.util.prompt_util import PromptUtil
from sentence_transformers.util import cos_sim
from app.util.sentence_util import SentenceUtil


class ModelUtil:
    def __init__(self):
        self.url = "${url}"
        self.authorization = "ACCESSCODE 647B1B5EE06D4413CF6E0459D2CC9616"
        self.content_type = "application/json"

    def get_user_question_response(self, question, refer_docs):
        # JSON数据
        json_data = {
            "model": "TelcomLLM",
            "messages": [
                {"role": "user", "content": "\"\"\"" + "。".join(refer_docs) + "\"\"\",基于上面文本回答，" + question}
            ],
            "temperature": 0.9,
            "top_p": 1,
            "penalty_score": 1
        }

        headers = {
            "Authorization": self.authorization,
            "Content-Type": self.content_type
        }

        # 发送POST请求，指定Content-Type为application/json
        response = requests.post(self.url, json=json_data, headers=headers)
        response_json = json.loads(response.text)

        answer = "没有找到答案"
        if response_json["CODE"] == "0000":
            answer = response_json["DATA"]["result"]

        return answer

    def get_cosine_similarity_matrix(self, answer_list):
        embeddings = SentenceUtil().model.encode(answer_list)
        similarity_matrix = cos_sim(embeddings, embeddings)
        return similarity_matrix

    def get_consistent_response(self, answer_list, vote_strategy, threshold):
        if vote_strategy == "cosine":
            similarity_matrix = self.get_cosine_similarity_matrix(answer_list)
            diagonal_indices = np.diag_indices_from(similarity_matrix)
            similarity_matrix[diagonal_indices] = np.nan
            average_similarity = np.nanmean(similarity_matrix)
            if average_similarity > threshold:
                return max(set(answer_list), key=answer_list.count)
        else:
            print("cannot provide reliable response")
            return "无法提供可靠回答"

    def get_compressed_user_question_with_consistency(self, question: str, vote_n: int, vote_strategy: str, threshold: float):
        df = pd.read_csv("C:\\Users\wangyy592\PycharmProjects\LLMDeepAppHybrid\prompts\summarize_tag.csv")
        prompt_util = PromptUtil()
        message = []
        prefix_token = "\n**任务提示**\n总结用户提出的问题\n"

        # generate care pattern prompt with prefix
        for i, row in df.iterrows():
            prompt_token = prompt_util.generate_prefix_prompt(
                prefix_token,
                "请对用户提出的问题“" + row['question'] + "”" +
                prompt_util.generate_tag_prompt(row['task'], row['action'], row['goal'], "**")
            )
            message.append({
                "role": "user",
                "content": prompt_token,
            })
            message.append({
                "role": "assistant",
                "content": row['answer'],
            })

        message.append({
            "role": "user",
            "content": prompt_util.generate_prefix_prompt(
                prefix_token,
                "假设你是产品运营人员，需要对用户提出的问题进行记录“" + question + "”" +
                prompt_util.generate_tag_prompt(
                    "重新描述用户提出的问题",
                    "先剔除礼貌用语、语气词。再识别用户问题的关键信息，重点关注名词、动词。然后找出用户都关注哪些问题。最后使用疑问句对问题进行重新表述",
                    "使用户提出的问题更简洁清晰、方便理解和分类",
                    "**")
            )
        })

        json_data = {
            "model": "TelcomLLM",
            "messages": message,
            "temperature": 0.6,
            "top_p": 0.6,
            "repetition_penalty": 1
        }

        headers = {
            "Authorization": self.authorization,
            "Content-Type": self.content_type
        }

        answer_list = []
        # 发送POST请求，指定Content-Type为application/json
        for i in range(0, vote_n):
            response = requests.post(self.url, json=json_data, headers=headers)
            response_json = json.loads(response.text)
            print(i, " ", response_json)
            try:
                answer = response_json["choices"][0]["message"]["content"].replace("<|im_end|>\n","")
                answer_list.append(answer)
            except Exception as e:
                return "模型服务异常：" + str(e)
        return self.get_consistent_response(answer_list, vote_strategy, threshold)

