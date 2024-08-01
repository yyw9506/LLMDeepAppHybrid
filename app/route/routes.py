# encoding:utf-8
from flask import Flask, Blueprint, jsonify, request
from app.service.service import get_compressed_question_from_model_with_self_consistency, \
    switch_model_domain, get_answer_from_model_with_self_consistency
import threading

main_blueprint = Blueprint('main', __name__)


@main_blueprint.route("/")
def index():
    return "[bot is on]"


@main_blueprint.route('/answerQuestion', methods=['POST'])
def answer_question():
    data_json = request.get_json()
    message = data_json["question"]
    return jsonify({
        "code": 0,
        "message": "",
        "data": get_answer_from_model_with_self_consistency(message, 3, "cosine", 0.6)
    })


@main_blueprint.route('/extractQuestion', methods=['POST'])
def compress_question():
    data_json = request.get_json()
    question = data_json["question"]
    return jsonify({
        "code": 0,
        "message": "",
        "data": get_compressed_question_from_model_with_self_consistency(question, 3, "cosine", 0.6)
    })


@main_blueprint.route('/switchDomain', methods=['POST'])
def switch_domain():
    data_json = request.get_json()
    domain = data_json["domain"]
    # init_learning_process = threading.Thread(target=switch_model_domain(domain))
    # init_learning_process.start()
    switch_model_domain(domain)
    return jsonify({
        "code": 0,
        "message": "Success",
        "data": None
    })


# 接口返回样例参数
@main_blueprint.route('/getHighFrequencyQuestions', methods=['GET'])
def get_high_frequency_questions():
    high_frequency_questions = [
        "社区指标加工口径", "社区指标周期", "省分账号如何赋权"
    ]
    return jsonify({
        "code": 0,
        "message": "",
        "data": high_frequency_questions
    })


# 接口返回样例参数
@main_blueprint.route('/getSimilarHistoryQuestions', methods=['GET'])
def get_similar_history_questions():
    similar_history_questions = [
        "社区指标加工口径", "社区指标周期", "省分账号如何赋权"
    ]
    return jsonify({
        "code": 0,
        "message": "",
        "data": similar_history_questions
    })


# 接口返回样例参数
@main_blueprint.route('/feedback_to_answer', methods=['POST'])
def feedback_to_answer():
    return jsonify({
        "code": 0,
        "message": "",
        "data": "感谢您的反馈！"
    })
