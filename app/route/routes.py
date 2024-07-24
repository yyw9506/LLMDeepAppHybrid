from flask import Blueprint, jsonify, request
from app.service.service import get_answer_from_model

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route("/")
def index():
    return "bot is on"

# 定义路由和视图函数
@main_blueprint.route('/answer',methods=['POST'])
def answer():
    data_json = request.get_json()
    message = data_json["message"]
    return get_answer_from_model(message)

