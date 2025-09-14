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
            # después de dicom_data = pydicom.dcmread(filepath)
            px = dicom_data.get("PixelSpacing", [0.1, 0.1])
            if isinstance(px, (list, pydicom.multival.MultiValue)) and len(px) >= 1:
                pixel_spacing_mm = float(px[0])
            else:
                pixel_spacing_mm = 0.1  # fallback seguro si el header no lo trae

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

            # antes: image_with_rois, rois = segment_and_draw_rois(image_array)
            image_with_rois, rois, angles = segment_and_draw_rois(image_array, pixel_spacing_mm=pixel_spacing_mm)

            # === DEBUG: dibujar ROIs (sin duplicar imports) ===
            def save_debug_rois(image, rois, out_path="images/rois_debug.png", angles=None):
                import cv2, os
                img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                colors = {
                    "attenuator_rois": (255, 0, 255),  # magenta
                    "background_roi": (0, 255, 255),   # amarillo
                    "roi_horizontal":  (0, 255, 0),    # verde (borde derecho)
                    "roi_vertical":    (0, 0, 255),    # rojo  (borde inferior)
                    "roi_small":       (255, 255, 0),  # celeste
                }
                for key, roi in rois.items():
                    if roi is None:
                        continue
                    x, y, w, h = roi
                    cv2.rectangle(img_color, (x, y), (x+w, y+h), colors.get(key, (255,255,255)), 2)
                    label = key
                    if angles and key in angles and angles[key] is not None:
                        label += f"  θ={angles[key]:.2f}°"
                    cv2.putText(img_color, label, (x, max(15, y-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(key, (255,255,255)), 1, cv2.LINE_AA)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, img_color)

            os.makedirs("images", exist_ok=True)
            save_debug_rois(image_array, rois, out_path="images/rois_debug.png", angles=angles)

            # === Normalización para SNR/SDNR/NNPS ===
            x_bg, y_bg, w_bg, h_bg = rois["background_roi"]
            roi_background = original_image[y_bg:y_bg+h_bg, x_bg:x_bg+w_bg]
            mean_background = float(np.mean(roi_background))
            normalized_image = original_image / (mean_background if mean_background != 0 else 1.0)

            # NNPS (opcional)
            roi_background_512 = extract_roi(roi_background, centered=True, size=(512, 512))
            calculate_nnps(roi_background_512, save_csv=True, csv_filename="nnps_output.csv")

            # === pMTF H (borde INFERIOR): usar función horizontal tal como la tenés ===
            roi_horiz = rois["roi_vertical"]  # borde inferior
            f_grid_h, pmtf_h, metrics_h, debug_h = pmft_horizontal_iaea_like(
                original_image,
                roi_horiz,
                pixel_spacing_mm=pixel_spacing_mm,
                angle_tilt_deg=angles.get("roi_vertical", 0.0),
                super_sampling_factor=10,
                hann_window_mm=25.0,
                bin_pitch=0.25,
                out_png="images/pMTF_horizontal_debug.png",
                smooth_esf="median5",
                smooth_pmtf=False
            )
            print("=== MÉTRICAS H (IAEA-like) ==="); print(metrics_h)

            # === pMTF V (borde DERECHO): reutilizar horizontal ROTANDO 90° y con ángulo 180-3.91 ===
            x, y, w, h = rois["roi_horizontal"]
            roi_v = original_image[y:y+h, x:x+w]
            roi_v_rot = np.rot90(roi_v, k=1)  # 90° CCW

            f_v, p_v, met_v, dbg_v = pmft_horizontal_iaea_like(
                image=roi_v_rot,
                roi_horizontal_edge=(0, 0, roi_v_rot.shape[1], roi_v_rot.shape[0]),
                pixel_spacing_mm=pixel_spacing_mm,
                angle_tilt_deg=angles.get("roi_horizontal", 0.0) - 90.0,
                super_sampling_factor=10,
                hann_window_mm=25.0,
                bin_pitch=0.25,
                out_png="images/pMTF_vertical_debug.png",
                smooth_esf="median5",
                smooth_pmtf=False
            )
            print("=== MÉTRICAS V (IAEA-like) ==="); print(met_v)



            os.makedirs("images", exist_ok=True)

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
                "pixel_spacing_mm": pixel_spacing_mm,
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
        

