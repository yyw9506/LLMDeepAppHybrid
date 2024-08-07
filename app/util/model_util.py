# encoding:utf-8
import requests
import json
import numpy as np
import pandas as pd
from app.util.prompt_util import PromptUtil
from app.util.sentence_util import SentenceUtil
from app.util.console_util import ConsoleUtil
from sentence_transformers.util import cos_sim


class ModelUtil:
    def __init__(self):
        self.url = "http://10.188.48.146:8088/naturalLanguageProcessing/llm-chat072402/v1/chat/completions"
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
                return (max(set(answer_list), key=answer_list.count)
                        .replace("Json字符串:", "")
                        .replace("Json字符串如下：", "")
                        .replace("```", "")
                        .replace("Json", "")
                        .replace("json", "")
                        .replace("**输出:**","")
                        .replace("****", "")
                )
        else:
            print("cannot provide reliable response")
            return "无法提供可靠回答"

    def get_answer_to_question_with_consistency(self, question: str, vote_n: int, vote_strategy: str,
                                                threshold: float, ref_indices: list):
        prompt_util = PromptUtil()
        df = pd.read_csv("C:\\Users\\wangyy592\\PycharmProjects\\LLMDeepApp\\prompts\\business_chat.csv")
        prefix_token = "\n**目的**\n回答用户的问题\n"
        message = []
        history = "历史问题：\n"
        counter = 1
        for i, row in df.iterrows():
            if i in ref_indices:
                append_info = (
                    "历史问题" + str(counter) + "：" + str(row["question"]) + "\n"
                    + "问题" + str(counter) + "提出方：" + str(row["question_source"]) + "\n"
                    + "问题" + str(counter) + "背景：" + str(row["background"]) + "\n"
                    + "问题" + str(counter) + "责任方：" + str(row["question_solver"]) + "\n"
                    + "问题" + str(counter) + "归属系统：" + str(row["question_to"]) + "\n"
                    + "问题" + str(counter) + "解决建议：" + str(row["advice"]) + "\n"
                    + "问题" + str(counter) + "运营同事最终回复：" + str(row["last_reply"]) + "\n\n\n"
                )
                history += append_info
                counter += 1
        history.replace("? ? ? ?", "")
        message.append({
            "role": "user",
            "content": prompt_util.generate_prefix_prompt(
                prefix_token,
                prompt_util.generate_chat_prompt(
                    character="假设你是某公司某软件项目组的运营人员，已经学习了下方的历史问题的业务和技术知识",
                    history=history,
                    ambition="现在请基于你对历史问题的学习和理解，对用户提出的问题”" + question + "“以Json的字符串的格式给出回答",
                    task="给出一个标准的Json字符串。Json字符串第一个字段是advice，请参考历史问题，在这个字段给出回答、可行的方案或解决问题的建议，并说明原因。第一个字段的长度不超过250字。"
                         "第二个字段是contact, 请在这个字段给出应该对问题负责的部门或系统。如果无法确定，请返回空字符串，不要返回虚构的人或部门，第二个字段的长度不超过20字",
                    spliter="**"
                )
            )
        })

        json_data = {
            "model": "TelcomLLM",
            "messages": message,
            "temperature": 0.3,
            "repetition_penalty": 1,
            "presence_penalty": -0.3,
            "logprobs": True
        }
        print("prompt: ", message)
        answer_list = []
        for i in range(0, vote_n):
            answer_list.append(self.get_response_from_llm(json_data))
            ConsoleUtil.print_progress(i + 1, vote_n, prefix='answer voting', suffix='complete', bar_length=50)

        try:
            answer_to_json = self.get_consistent_response(answer_list, vote_strategy, threshold)
            print("answer", answer_to_json)
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
                "假设你是某公司某软件项目组的运营人员，请按照下方要求处理用户提出的问题“" + row['question'] + "”" +
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
                "假设你是某公司某软件项目组的运营人员，请按照下方要求处理用户提出的问题：“" + question + "”\n" +
                prompt_util.generate_tag_prompt(
                    "用一句话重新描述用户提出的问题, 长度120字以内，否则将收到惩罚",
                    "先剔除礼貌用语、语气词。再识别用户问题的关键信息，重点关注名词、动词。然后找出用户都关注哪些问题。最后使用疑问句对问题进行重新表述",
                    "使用户提出的问题更简洁清晰、方便理解和分类",
                    "**")
            )
        })

        json_data = {
            "model": "TelcomLLM",
            "messages": message,
            "temperature": 0.6,
            "repetition_penalty": 1,
            "presence_penalty": -0.5
        }

        print("prompt: ", message)
        answer_list = []
        for i in range(0, vote_n):
            answer_list.append(self.get_response_from_llm(json_data))
            ConsoleUtil.print_progress(i + 1, vote_n, prefix='extract voting', suffix='complete', bar_length=50)
        return self.get_consistent_response(answer_list, vote_strategy, threshold)
