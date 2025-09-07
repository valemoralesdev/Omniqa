
from pymongo import MongoClient
from app.config import Config

# Conectar con MongoDB usando la URI definida en config.py
client = MongoClient(Config.MONGO_URI)

# Acceder a las bases de datos específicas
api_config_db = client["omniqa_config"]         # Configuración general de la API
quality_db = client["TeroDB"]                   # Resultados de calidad (QA)
study_results_db = client["processings"]        # Resultados de estudios específicos

# Acceso directo a la colección 'ct' de la base 'TeroDB'
ct_collection = client["TeroDB"]["ct"]

def get_database(db_name):
    """
    Retorna la base de datos solicitada dentro del servidor MongoDB.
    :param db_name: Nombre de la base de datos (api_config, quality_control, study_results)
    :return: Instancia de la base de datos en MongoDB
    """
    return client[db_name]