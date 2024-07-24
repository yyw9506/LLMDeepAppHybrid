from app.util.model_util import ModelUtil


if __name__ == "__main__":
    http = ModelUtil()
    question = "执行人维护页面可以做什么操作"
    lines = [
        "代理派单人是实现对其关联的任一个人员角色是派单人的所有下级执行人下发任务",
        "人员角色为代理派单人是在【代理派单人维护】页面进行人员维护",
        "人员角色为派单人是在【执行人维护】页面进行人员维护",
        "代理派单人是在派单人角色的编辑页面内由字段“代理派单人”进行关联",
        "代理派单人的区域要求是派单人区域的同级或者下级区域，则派单人可成功关联该代理派单人"
    ]

    http.get_response(question, lines)