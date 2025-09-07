from flask import Blueprint, request, jsonify, send_file, current_app
import os
import cv2
import zipfile
from services.dicom_service import process_dicom_image, process_dicom_series_from_folder
from database.mongo_connection import ct_collection
from utils.dicom_utils import extract_dicom_metadata  # Debe usar to_json_dict internamente

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
def upload_dicom():
    """Carga un archivo DICOM o ZIP de serie, extrae el Pixel Data y lo procesa"""

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = file.filename

    # Obtener rutas desde Flask config
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    result_folder = current_app.config["RESULT_FOLDER"]

    # Guardar archivo en uploads/
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Procesamiento basado en extensión
    if filename.lower().endswith(".zip"):
        extract_folder = os.path.join(upload_folder, os.path.splitext(filename)[0])
        os.makedirs(extract_folder, exist_ok=True)

        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        except zipfile.BadZipFile:
            return jsonify({"error": "Uploaded file is not a valid ZIP archive"}), 400

        result, error = process_dicom_series_from_folder(extract_folder)

    elif filename.lower().endswith(".dcm"):
        result, error = process_dicom_image(filepath)

    else:
        return jsonify({"error": "Unsupported file type"}), 400

    if error:
        return jsonify({"error": error}), 400

    # Guardar imagen procesada
    if result.get("image") is not None:
        output_path = os.path.join(result_folder, f"{os.path.splitext(filename)[0]}_processed.png")
        cv2.imwrite(output_path, result["image"])
        result["image_path"] = output_path

    # Quitar imagen cruda antes de guardar en DB
    result.pop("image", None)

    # Determinar archivo DICOM para extraer metadata
    dicom_metadata = {}
    dicom_path = filepath  # por defecto es el archivo subido

    if filename.lower().endswith(".zip"):
        # Buscar primer DICOM dentro del ZIP extraído
        dicom_files = [
            os.path.join(extract_folder, f) for f in os.listdir(extract_folder)
            if f.lower().endswith(".dcm")
        ]
        if dicom_files:
            dicom_path = dicom_files[0]
        else:
            dicom_path = None

    # Extraer metadata si se tiene un path válido
    if dicom_path:
        try:
            dicom_metadata = extract_dicom_metadata(dicom_path)
        except Exception as e:
            dicom_metadata = {"error": f"Failed to extract DICOM metadata: {str(e)}"}

    result["dicom_metadata"] = dicom_metadata

    try:
        inserted = ct_collection.insert_one(result)
        result["_id"] = str(inserted.inserted_id)
    except Exception as db_error:
        return jsonify({
            "message": "Processed, but failed to store in database",
            "details": result,
            "db_error": str(db_error)
        }), 207

    return jsonify({
        "message": "Processed",
        "details": result
    }), 200
