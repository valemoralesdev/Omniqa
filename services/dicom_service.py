from processing_tools.processing_factory import ProcessingFactory
import pydicom
import os

def process_dicom_image(filepath):
    try:
        dicom_data = pydicom.dcmread(filepath)
        modality = dicom_data.get("Modality", "").strip().lower()
        study_description = dicom_data.get("StudyDescription", "").strip().lower()

        print(f"Estudio recibido: Modality={modality}, StudyDescription={study_description}")
        processing_algorithm = ProcessingFactory.get_algorithm(modality, study_description)

        if processing_algorithm is None:
            return None, "No processing algorithm available"

        result = processing_algorithm.process(filepath)

        if not isinstance(result, dict) or "image" not in result:
            return None, "Invalid algorithm output format"

        return result, None

    except Exception as e:
        print(f"Error en el procesamiento: {e}")
        return None, str(e)

def process_dicom_series_from_folder(folder_path):
    """
    Procesa una serie DICOM extraída de un ZIP.

    Args:
        folder_path (str): Ruta a la carpeta con los archivos .dcm

    Returns:
        tuple: (result_dict, error_message)
    """

    # Buscar todos los .dcm en la carpeta
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]

    if not dicom_files:
        return None, "No DICOM files found in uploaded ZIP."

    # Usar el primer DICOM como referencia para decidir algoritmo
    try:
        ref_dicom = pydicom.dcmread(dicom_files[0])
    except Exception as e:
        return None, f"Error reading reference DICOM: {str(e)}"

    modality = (ref_dicom.get("Modality") or "").strip().lower()
    study_description = (ref_dicom.get("StudyDescription") or "").strip().lower()

    print(f"Serie recibida: Modality={modality}, StudyDescription={study_description}")
    processing_algorithm = ProcessingFactory.get_algorithm(modality, study_description)

    if processing_algorithm is None:
        return None, "No processing algorithm available for this series."

    # Ejecutar procesamiento
    try:
        result = processing_algorithm.process(dicom_files[0])  # Se pasa un archivo, el algoritmo maneja el resto
    except Exception as e:
        return None, f"Error during processing: {str(e)}"

    if not isinstance(result, dict) or "image" not in result:
        return None, "Invalid algorithm output format from series processing"

    return result, None
