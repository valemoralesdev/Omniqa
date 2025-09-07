import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class Config:
    """Configuración general de la API."""
    FLASK_APP = os.getenv("FLASK_APP", "app.py")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")

    # Configuración de MongoDB
    MONGO_URI = os.getenv("MONGO_URI")

    # Configuración de Flask
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    TESTING = os.getenv("TESTING", "False").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")

    # Directorios de almacenamiento
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Ruta base del proyecto
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "../data/dicom_files/"))
    RESULT_FOLDER = os.getenv("RESULT_FOLDER", os.path.join(BASE_DIR, "../data/processed/"))

    # Crear directorios si no existen
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
