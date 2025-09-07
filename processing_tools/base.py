from abc import ABC, abstractmethod

class ProcessingAlgorithm(ABC):
    """
    Clase base para algoritmos de procesamiento.
    Todos los algoritmos deben heredar de esta clase e implementar `process`.
    """

    @abstractmethod
    def process(self, filepath) -> dict:
        """
        Método abstracto que debe ser implementado en cada algoritmo.
        :param filepath: Ruta del archivo DICOM a procesar.
        :return: Imagen procesada y ROIs extraídas.
        """
        pass
