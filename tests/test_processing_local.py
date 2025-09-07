import os
from processing_tools.processing_factory import ProcessingFactory
import pydicom
import cv2

# Ruta al archivo DICOM que querés testear
dicom_path = "data/uploads/mamografia.dcm"

# Leer metadata
dicom = pydicom.dcmread(dicom_path)
modality = dicom.get("Modality", "").lower().strip()
study_description = dicom.get("StudyDescription", "").lower().strip()

print(f"🧪 Modality: {modality}, Study Description: {study_description}")

# Obtener algoritmo desde la fábrica
algorithm = ProcessingFactory.get_algorithm(modality, study_description)

if algorithm is None:
    print("⚠️ No hay algoritmo para este estudio.")
    exit()

# Ejecutar procesamiento
result = algorithm.process(dicom_path)

# Guardar resultado si hay imagen
if result["image"] is not None:
    output_path = f"data/results/test_{os.path.splitext(os.path.basename(dicom_path))[0]}.png"
    cv2.imwrite(output_path, result["image"])
    print(f"✅ Imagen procesada guardada en: {output_path}")

# Mostrar log, métricas u otros resultados
if result.get("log"):
    print("Log:", result["log"])
if result.get("metrics"):
    print("Métricas:")
    for key, val in result["metrics"].items():
        print(f"  - {key}: {val}")
if result.get("rois"):
    print(f"Total ROIs detectadas: {len(result['rois'])}")
