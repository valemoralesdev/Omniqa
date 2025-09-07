from flask import Blueprint
from controllers.upload_controller import upload_bp  # Importamos el controlador

# Crear Blueprint principal de la API
api_bp = Blueprint('api', __name__)

# Registrar los Blueprints individuales
api_bp.register_blueprint(upload_bp, url_prefix="/upload")
