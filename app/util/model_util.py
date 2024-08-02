# encoding:utf-8
import requests
import json
import numpy as np
import pandas as pd
from app.util.prompt_util import PromptUtil
from sentence_transformers.util import cos_sim
from app.util.sentence_util import SentenceUtil
from app.util.console_util import ConsoleUtil


class ModelUtil:
    def __init__(self):
        self.url = ""
        self.authorization = "ACCESSCODE 647B1B5EE06D4413CF6E0459D2CC9616"
        self.content_type = "application/json"

    def get_response_from_llm(self, json_data):
        headers = {
            "Authorization": self.authorization,
            "Content-Type": self.content_type
        }
        response = requests.post(self.url, json=json_data, headers=headers)
        response_json = json.loads(response.text)
        try:
            answer = (
                response_json["choices"][0]["message"]["content"]
                .replace("<|im_end|>", "")
                .replace("<|im_start|>", "")
                .replace("\n", "")
                .replace("upyter", "")
            )
            return answer
        except Exception as e:
            return "模型服务异常：" + str(e)

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
                return max(set(answer_list), key=answer_list.count).replace("Json字符串：```", "")
        else:
            print("cannot provide reliable response")
            return "无法提供可靠回答"

    def get_answer_to_question_with_consistency(self, question: str, vote_n: int, vote_strategy: str,
                                                threshold: float, ref_indices: list):
        prompt_util = PromptUtil()
        df = pd.read_csv("C:\\Users\\wangyy592\\PycharmProjects\\LLMDeepApp\\prompts\\business_chat.csv")
        prefix_token = "\n**目的**\n提供问题的解决方案和建议\n"
        message = []
        history = "以下是可供参考的历史问题：\n"
        counter = 1
        for i, row in df.iterrows():
            if i in ref_indices:
                append_info = (
                    "历史问题" + str(counter) + "：" + str(row["question"]) + "\n"
                    + "问题" + str(counter) + "提出方：" + str(row["question_source"]) + "\n"
                    + "问题" + str(counter) + "背景：" + str(row["background"]) + "\n"
                    + "问题" + str(counter) + "责任方：" + str(row["question_to"]) + "\n"
                    + "问题" + str(counter) + "解决建议：" + str(row["advice"]) + "\n"
                    + "问题" + str(counter) + "最终回复：" + str(row["last_reply"]) + "\n\n\n"
                )
                history += append_info
                counter += 1

        message.append({
            "role": "user",
            "content": prompt_util.generate_prefix_prompt(
                prefix_token,
                prompt_util.generate_chat_prompt(
                    character="假设你是某公司某软件项目组的运营人员，熟悉项目组的业务和技术",
                    history=history,
                    ambition="现在你需要根据上面已有的背景知识，对用户提出的问题”" + question + "“以固定的格式给出解决方案和建议",
                    task="返回一个Json字符串。第一个字段是advice，请在这个字段给出你的解决方案和建议，并解释原因，长度不超过250字。第二个字段是people, "
                    "请在这个字段给出你确定能够解决用户问题的人和联系方式或部门名称。如果无法确定，请返回空字符串，不要返回虚构的人和联系方式，长度不超过50字",
                    spliter="**"
                )
            )
        })

        json_data = {
            "model": "TelcomLLM",
            "messages": message,
            "temperature": 0.5,
            "top_p": 0.4,
            "repetition_penalty": 1
        }

        answer_list = []
        for i in range(0, vote_n):
            answer_list.append(self.get_response_from_llm(json_data))
            ConsoleUtil.print_progress(i + 1, vote_n, prefix='answer voting', suffix='complete', bar_length=50)

        try:
            answer_to_json = self.get_consistent_response(answer_list, vote_strategy, threshold)
            return json.loads(answer_to_json)
        except Exception as e:
            return {
                "exception": str(e)
            }

    def get_compressed_question_with_consistency(self, question: str, vote_n: int, vote_strategy: str,
                                                 threshold: float):
        prompt_util = PromptUtil()
        df = pd.read_csv("C:\\Users\wangyy592\PycharmProjects\LLMDeepAppHybrid\prompts\summarize_tag.csv")
        prefix_token = "\n**目的**\n一句话总结用户提出的问题\n"
        message = []

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
                    "用一句话重新描述用户提出的问题, 长度120字以内",
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

        answer_list = []
        for i in range(0, vote_n):
            answer_list.append(self.get_response_from_llm(json_data))
            ConsoleUtil.print_progress(i + 1, vote_n, prefix='extract voting', suffix='complete', bar_length=50)
        return self.get_consistent_response(answer_list, vote_strategy, threshold)
