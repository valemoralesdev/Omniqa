import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
import cv2
import numpy as np
import cv2
from scipy.ndimage import rotate, gaussian_filter1d
from sklearn.linear_model import LinearRegression

def calculate_snr(image, roi_signal):
    x, y, w, h = roi_signal
    signal_region = image[y:y+h, x:x+w]
    mean_signal = np.mean(signal_region)
    std_signal = np.std(signal_region)
    if std_signal == 0:
        return np.nan
    return mean_signal / std_signal

def calculate_sdnr(image, roi_signal, roi_background):
    x_s, y_s, w_s, h_s = roi_signal
    x_b, y_b, w_b, h_b = roi_background
    signal_region = image[y_s:y_s+h_s, x_s:x_s+w_s]
    background_region = image[y_b:y_b+h_b, x_b:x_b+w_b]
    mean_signal = np.mean(signal_region)
    mean_background = np.mean(background_region)
    std_background = np.std(background_region)
    if std_background == 0:
        return np.nan
    return np.abs(mean_background - mean_signal) / std_background

import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
import cv2
from scipy.ndimage import rotate, gaussian_filter1d
from sklearn.linear_model import LinearRegression

def calcular_mtf_desde_roi(roi_coords, original_image, pixel_spacing=0.1, super_sampling_factor=4):
    x, y, w, h = roi_coords
    roi = original_image[y:y+h, x:x+w]

    # Detectar ángulo del borde
    roi_uint8 = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(roi_uint8, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None, None, None

    main_contour = max(contours, key=lambda c: len(c))
    pts = main_contour.squeeze()
    xs, ys = pts[:, 0], pts[:, 1]
    model = LinearRegression().fit(xs.reshape(-1, 1), ys)
    angle_deg = np.rad2deg(np.arctan(model.coef_[0]))

    # Rotar ROI para alinearlo con el borde
    angle_correction = angle_deg - 90
    print(angle_correction)
    roi_rotated = rotate(roi, angle_correction, reshape=True, order=3)

    # Recortar una región más grande para supermuestreo
    crop_w, crop_h = int(roi_rotated.shape[1] * 0.9), int(roi_rotated.shape[0] * 0.9)
    cropped_roi = roi_rotated[roi_rotated.shape[0]//2 - crop_h//2:roi_rotated.shape[0]//2 + crop_h//2,
                              roi_rotated.shape[1]//2 - crop_w//2:roi_rotated.shape[1]//2 + crop_w//2]

    # Crear ESF supermuestreada
    esf_length = cropped_roi.shape[1] * super_sampling_factor
    esf = np.zeros(esf_length)
    counts = np.zeros(esf_length)

    for i in range(cropped_roi.shape[0]):
        offset = (i / cropped_roi.shape[0]) * super_sampling_factor
        for j in range(cropped_roi.shape[1]):
            esf_idx = int(j * super_sampling_factor + offset)
            esf[esf_idx] += cropped_roi[i, j]
            counts[esf_idx] += 1

    esf /= np.maximum(counts, 1)
    esf = gaussian_filter1d(esf, sigma=0.5 * super_sampling_factor)

    # Calcular LSF
    lsf = np.gradient(esf)
    lsf /= np.max(np.abs(lsf))

    
    # FFT
    window = np.blackman(len(lsf))
    lsf_windowed = lsf * window
    N_fft = len(lsf_windowed)
    freqs = fftfreq(N_fft, d=pixel_spacing/super_sampling_factor)[:N_fft // 2]
    mtf = np.abs(fft(lsf_windowed))[:N_fft // 2]
    mtf /= mtf[0]

    # Interpolación para frecuencias MTF específicas
    interp_mtf = interp1d(mtf, freqs, bounds_error=False, fill_value="extrapolate")
    mtf_vals = {
        "MTF@50%": round(float(interp_mtf(0.5)), 3),
        "MTF@20%": round(float(interp_mtf(0.2)), 3),
        "MTF@10%": round(float(interp_mtf(0.1)), 3),
    }

    return freqs, mtf, mtf_vals
