from flask import Flask
from .config import Config
from .db import init_db
from .routes_user import bp_user
from .routes_manager import bp_manager

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config())

    # ensure instance folder exists + db tables exist
    init_db(app)

    # blueprints
    app.register_blueprint(bp_user)
    app.register_blueprint(bp_manager)

    return app
