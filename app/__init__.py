from flask import Flask
from app.config import Config
from app.extensions import mongo, jwt, cors

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Inicializar extensiones
    mongo.init_app(app)
    jwt.init_app(app)
    cors.init_app(app)

    # Registrar Blueprints
    from api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
