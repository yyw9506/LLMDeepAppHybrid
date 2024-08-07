# encoding:utf-8
from flask import Flask
from app.route.routes import main_blueprint


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main_blueprint)
    return app
