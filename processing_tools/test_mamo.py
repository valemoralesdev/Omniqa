import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_tools.atia_processing_mamo import AtiaProcessingMamo

# Ruta al archivo DICOM local (modificá esta ruta)
dicom_path = "C:/Users/valen/procesamiento/OmniQA_api-main/OmniQA_api-main/images/5.dcm"

# Instanciar la clase y ejecutar procesamiento
processor = AtiaProcessingMamo()
result = processor.process(dicom_path)

# Mostrar métricas por consola
print("\n=== RESULTADO ===")
print(result["metrics"])
