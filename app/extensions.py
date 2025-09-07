from flask_pymongo import PyMongo
from flask_cors import CORS
from flask_jwt_extended import JWTManager

# Inicializar extensiones
mongo = PyMongo()
jwt = JWTManager()
cors = CORS()
