# encoding:utf-8
import requests
import json


class ModelUtil:
    def __init__(self):
        self.url = "http://10.125.128.9:39391/naturalLanguageProcessing/unicom-llm/chat"
        self.authorization = "ACCESSCODE 3D902721BFDF47D1CCB8E5AAE89C26BD"
        self.content_type = "application/json"

    def get_response(self, question, referDocuments):
        # JSON数据
        json_data = {
            "messages": [
                {"role": "user", "content": "\"\"\"" + "。".join(referDocuments) + "\"\"\",基于上面文本回答，" + question}
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
        if (response_json["CODE"] == "0000"):
            answer = response_json["DATA"]["result"]

        print(answer)
        return answer

