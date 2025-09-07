from processing_tools.base import ProcessingAlgorithm
import cv2
from processing_tools.metrics import *
from processing_tools.image_processing import segment_and_draw_rois
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
from processing_tools.nnps import extract_roi, calculate_nnps
from pymongo import MongoClient
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
from processing_tools.detectability import calculate_dprime
from decimal import Decimal, ROUND_HALF_UP


class AtiaProcessingMamo(ProcessingAlgorithm):
    def process(self, filepath):
        try:
            dicom_data = pydicom.dcmread(filepath)
            image_array = dicom_data.pixel_array.astype(np.float32)
            original_image = image_array.copy()

            window_center = dicom_data.get("WindowCenter", np.median(image_array))
            window_width = dicom_data.get("WindowWidth", np.max(image_array) - np.min(image_array))
            if isinstance(window_center, (list, pydicom.multival.MultiValue)):
                window_center = float(window_center[0])
            if isinstance(window_width, (list, pydicom.multival.MultiValue)):
                window_width = float(window_width[0])
            min_window = window_center - (window_width / 2)
            max_window = window_center + (window_width / 2)
            image_array = np.clip(image_array, min_window, max_window)
            image_array = ((image_array - min_window) / (max_window - min_window)) * 255.0
            image_array = image_array.astype(np.uint8)

            image_with_rois, rois = segment_and_draw_rois(image_array)
            
            x_bg, y_bg, w_bg, h_bg = rois["background_roi"]
            roi_background = original_image[y_bg:y_bg+h_bg, x_bg:x_bg+w_bg]
            mean_background = np.mean(roi_background)
            normalized_image = original_image / mean_background

            x_sig, y_sig, w_sig, h_sig = rois["roi_small"]
            roi_signal = original_image[y_sig:y_sig+h_sig, x_sig:x_sig+w_sig]
    
            x, y, w, h = rois["roi_horizontal"]
            roi_horizontal_array = original_image[y:y+h, x:x+w]
            a, b, c, d = rois["roi_vertical"]
            roi_vertical_array = original_image[b:b+d, a:a+c]

            roi_background_512 = extract_roi(roi_background, centered=True, size=(512, 512))
            calculate_nnps(roi_background_512, save_csv=True, csv_filename="nnps_output.csv")
                # --- Calcular MTFs ---
            # Si querés usar un ángulo fijo de 2.5°
            
            freqs_h, mtf_h, mtf_vals_h = calcular_mtf_desde_roi((x, y, w, h), original_image, pixel_spacing=0.1, super_sampling_factor=4)
            freqs_v, mtf_v, mtf_vals_v = calcular_mtf_desde_roi((a, b, c, d), original_image, pixel_spacing=0.1, super_sampling_factor=4)


            os.makedirs("images", exist_ok=True)

            d1 = calculate_dprime(freqs_h, mtf_h, freqs_v, mtf_v, roi_background, roi_signal)

            print(f"\n🔍 Índice de detectabilidad (d'):")
            for key, value in d1.items():
                print(f"  {key}: {value}")
                
            def round_decimal(value, decimals=2):
                q = Decimal("1." + "0" * decimals)  # Ej: Decimal('1.00') para dos decimales
                return Decimal(float(value)).quantize(q, rounding=ROUND_HALF_UP)
            
                        # Obtener campos DICOM
            study_date = dicom_data.get("StudyDate", "")         # '20240524'
            study_time = dicom_data.get("StudyTime", "000000")   # '163052'

            # Normalizar y combinar
            study_datetime_str = study_date + study_time[:6]     # '20240524163052'
            fecha = datetime.strptime(study_datetime_str, "%Y%m%d%H%M%S")

            # Calcular métricas finales
            metrics = {
                "snr_signal": float(round_decimal(calculate_snr(normalized_image, rois["background_roi"]))),
                "sdnr": float(round_decimal(calculate_sdnr(normalized_image, rois["roi_small"], rois["background_roi"]))),
                "pixel_spacing_mm": 0.1,
                "mtf_horizontal": mtf_vals_h,
                "mtf_vertical": mtf_vals_v,
            }

            import json

            with open("images/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            # Guardar en MongoDB sin duplicar
            try:
                documento = {
                    "sop_instance_uid": dicom_data.SOPInstanceUID,
                    "fecha": fecha,
                    "tipo_estudio": dicom_data.get("StudyDescription", "desconocido"),
                    "origen_equipo": dicom_data.get("ManufacturerModelName", "desconocido"),
                    "archivo_dicom": os.path.basename(filepath),
                    "metricas": metrics,
                }

                client = MongoClient("mongodb://localhost:27017/")
                db = client["quality_control"]
                collection = db["results"]
                collection.create_index("sop_instance_uid", unique=True)
                
                # Insertar solo si no existe, o actualizar si ya está
                collection.update_one(
                    {"sop_instance_uid": documento["sop_instance_uid"]},  # criterio único
                    {"$set": documento},
                    upsert=True
                )

                print("✅ Documento actualizado/insertado en MongoDB.")
            except Exception as mongo_error:
                print("❌ Error guardando en MongoDB:", mongo_error)


            return {
                "image": image_with_rois,
                "rois": rois,
                "metrics": metrics
            }

        except Exception as e:
            print(f"❌ Error en ATIA Processing Mamo: {e}")
            return {"image": None, "rois": {}, "metrics": {}}
        

