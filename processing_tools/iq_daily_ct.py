import os
import numpy as np
import pydicom
import cv2
from processing_tools.base import ProcessingAlgorithm

class IqDailyCt(ProcessingAlgorithm):
    """
    Procesamiento de fantoma de calidad diaria en CT sobre series completas.
    Calcula métricas slice por slice y las promedia.
    """

    def process(self, filepath: str) -> dict:
        folder = os.path.dirname(filepath)
        dicom_files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".dcm")
        ])

        if not dicom_files:
            raise ValueError("No se encontraron archivos DICOM en la carpeta.")

        metrics_per_slice = []
        visual_slices = []

        for file in dicom_files:
            dicom = pydicom.dcmread(file)
            raw = dicom.pixel_array.astype(np.float32)

            slope = float(dicom.get('RescaleSlope', 1.0))
            intercept = float(dicom.get('RescaleIntercept', 0.0))
            dicom_image = raw * slope + intercept

            visual_image = np.clip(dicom_image, -500, 1500)
            visual_image = ((visual_image + 500) / 2000 * 255).astype(np.uint8)
            visual_slices.append(visual_image)

            _, thresh = cv2.threshold(visual_image, 40, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            max_radius = min(visual_image.shape) // 2 - 5
            radius = min(radius, max_radius)

            instance_uid = dicom.SOPInstanceUID
            slice_metrics = {
                "instance_uid": instance_uid
            }

            for scale in [0.25, 0.5, 0.75]:
                mask = np.zeros_like(visual_image, dtype=np.uint8)
                cv2.circle(mask, center, int(radius * scale), 255, -1)
                region_pixels = dicom_image[mask == 255]
                mean_val = float(np.mean(region_pixels))
                std_val = float(np.std(region_pixels))
                slice_metrics[f'mean_cfov_{int(scale*100)}'] = mean_val
                slice_metrics[f'std_cfov_{int(scale*100)}'] = std_val

            try:
                slice_metrics['uniformity_index'] = 1 - abs(
                    slice_metrics['mean_cfov_25'] - slice_metrics['mean_cfov_75']
                ) / slice_metrics['mean_cfov_50']
            except ZeroDivisionError:
                slice_metrics['uniformity_index'] = None

            metrics_per_slice.append(slice_metrics)

        if not metrics_per_slice:
            raise ValueError("No se pudieron calcular métricas en ningún slice.")

        # Promediar métricas
        mean_cfov_25 = np.mean([m['mean_cfov_25'] for m in metrics_per_slice])
        std_cfov_25 = np.mean([m['std_cfov_25'] for m in metrics_per_slice])
        mean_cfov_50 = np.mean([m['mean_cfov_50'] for m in metrics_per_slice])
        std_cfov_50 = np.mean([m['std_cfov_50'] for m in metrics_per_slice])
        mean_cfov_75 = np.mean([m['mean_cfov_75'] for m in metrics_per_slice])
        std_cfov_75 = np.mean([m['std_cfov_75'] for m in metrics_per_slice])
        uniformity_index = np.mean([
            m['uniformity_index'] for m in metrics_per_slice if m['uniformity_index'] is not None
        ])

        series_metrics = {
            "series_instance_uid": dicom.SeriesInstanceUID,
            "mean_cfov_25": mean_cfov_25,
            "std_cfov_25": std_cfov_25,
            "mean_cfov_50": mean_cfov_50,
            "std_cfov_50": std_cfov_50,
            "mean_cfov_75": mean_cfov_75,
            "std_cfov_75": std_cfov_75,
            "uniformity_index": uniformity_index,
            "total_slices": len(metrics_per_slice)
        }

        volume = np.stack(visual_slices)
        mid_slice = volume[len(visual_slices) // 2]

        _, thresh = cv2.threshold(mid_slice, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        max_radius = min(mid_slice.shape) // 2 - 5
        radius = min(radius, max_radius)

        output_image = cv2.cvtColor(mid_slice, cv2.COLOR_GRAY2BGR)
        cv2.circle(output_image, center, radius, (0, 255, 0), 2)
        for scale in [0.25, 0.5, 0.9]:
            cv2.circle(output_image, center, int(radius * scale), (255, 0, 0), 1)

        ref_dicom = pydicom.dcmread(dicom_files[0])
        log_text = (
            f"Fecha: {ref_dicom.get('StudyDate', 'N/A')}, "
            f"Descripción del Estudio: {ref_dicom.get('StudyDescription', 'N/A')}, "
            f"Equipo: {ref_dicom.get('Manufacturer', 'N/A')} {ref_dicom.get('ManufacturerModelName', 'N/A')}"
        )

        return {
            "image": output_image,
            "series_metrics": series_metrics,
            "slice_metrics": metrics_per_slice,
            "rois": [],
            "log": log_text,
            "equipment_name": ref_dicom.get('ManufacturerModelName', 'N/A'),
            "study_description": ref_dicom.get('StudyDescription', 'N/A'),
            "study_date": ref_dicom.get('StudyDate', 'N/A')
        }