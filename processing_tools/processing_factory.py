from processing_tools.atia_processing_mamo import AtiaProcessingMamo

class ProcessingFactory:
    """
    Fábrica de algoritmos de procesamiento.
    Determina qué algoritmo aplicar en función del tipo de imagen y estudio.
    """

    @staticmethod
    def get_algorithm(modality, study_description):
        """
        Devuelve la instancia del algoritmo adecuado según la modalidad y descripción del estudio.
        """
        modality = modality.lower()
        study_description = study_description.lower()

        # Asignación del algoritmo ATIA para mamografía
        if modality == "mg" and "" in study_description:
            return AtiaProcessingMamo()  # Se devuelve una instancia del algoritmo
        else:
            print(f"⚠️ Rechazo esperado: No hay un algoritmo asignado para {modality} - {study_description}")
            return None