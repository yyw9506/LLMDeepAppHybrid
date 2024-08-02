# encoding:utf-8
class PromptUtil:

    def __init__(self):
        pass

    def generate_broke_prompt(self, background: str, role: str, objectives: str, key_results: str, evolve: str, spliter: str):
        return (
            spliter + "_背景:_" + spliter + "\n" + background + "\n"
            + spliter + "_角色:_" + spliter + "\n" + role + '\n'
            + spliter + "_目标:_" + spliter + "\n" + objectives + "\n"
            + spliter + "_结果:_" + spliter + "\n" + key_results + "\n"
            + spliter + "_发展:_" + spliter + "\n" + evolve
        )

    def generate_care_prompt(self, context: str, action: str, result: str, example: str, spliter: str):
        return (
            spliter + "上下文" + spliter + "\n" + context + "\n"
            + spliter + "行动" + spliter + "\n" + action + "\n"
            + spliter + "结果" + spliter + "\n" + result + "\n"
            + spliter + "示例" + spliter + "\n" + example
         )

    def generate_chat_prompt(self, character: str, history: str, ambition: str, task: str, spliter: str):
        return (
            spliter + "你的角色：" + spliter + "\n" + character + "\n"
            + spliter + "背景知识：" + spliter + "\n" + history + "\n"
            + spliter + "目标:" + spliter + "\n" + ambition + "\n"
            + spliter + "任务:" + spliter + "\n" + task
         )

    def generate_tag_prompt(self, task: str, action: str, goal: str, spliter: str):
        return (
                spliter + "执行以下操作" + spliter + "\n" + task + "\n"
                + spliter + "在你执行操作时，请遵循以下指导" + spliter + "\n" + action + "\n"
                + spliter + "你的目标是" + spliter + "\n" + goal
        )

    def generate_prefix_prompt(self, prefix: str, token: str):
        return prefix + "\n" + token


