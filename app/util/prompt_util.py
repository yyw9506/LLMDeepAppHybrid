# encoding:utf-8
class PromptUtil:

    def __init__(self):
        pass

    def generate_broke_prompt(self, background: str, role: str, objectives: str, key_results: str, evolve: str, spliter: str):
        return (
            spliter + "背景" + spliter + "\n" + background + "\n"
            + spliter + "你的角色是" + spliter + "\n" + role + '\n'
            + spliter + "目标是" + spliter + "\n" + objectives + "\n"
            + spliter + "关键是" + spliter + "\n" + key_results + "\n"
            + spliter + "优化" + spliter + "\n" + evolve
        )

    def generate_care_prompt(self, context: str, action: str, result: str, example: str, spliter: str):
        return (
            spliter + "上下文" + spliter + "\n" + context + "\n"
            + spliter + "行动" + spliter + "\n" + action + "\n"
            + spliter + "结果" + spliter + "\n" + result + "\n"
            + spliter + "示例" + spliter + "\n" + example
         )


    def generate_tag_prompt(self, task: str, action: str, goal: str, spliter: str):
        return (
                spliter + "执行以下操作" + spliter + "\n" + task + "\n"
                + spliter + "在你执行操作时，请遵循以下指导" + spliter + "\n" + action + "\n"
                + spliter + "你的目标是" + spliter + "\n" + goal
        )

    def generate_prefix_prompt(self, prefix: str, token: str):
        return prefix + "\n" + token


