import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from scipy import signal

def flatten_image(image):
    """ Ajuste y sustracción de una superficie plana (flatten) """
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)
    A = np.c_[X.ravel(), Y.ravel(), np.ones_like(X.ravel())]
    coeffs, _, _, _ = np.linalg.lstsq(A, image.ravel(), rcond=None)
    plane = (coeffs[0] * X + coeffs[1] * Y + coeffs[2])
    return image - plane

def extract_roi(image, top_left=None, size=(512, 512), centered=False):
    h, w = size

    if centered or top_left is None:
        center_y, center_x = np.array(image.shape) // 2
        half_h = h // 2
        half_w = w // 2
        y_start = max(center_y - half_h, 0)
        x_start = max(center_x - half_w, 0)
    else:
        y_start, x_start = top_left

    roi = image[y_start:y_start + h, x_start:x_start + w]

    return roi

def calculate_nnps(image, roi_size=256, large_roi_size=512, pixel_spacing_mm=0.1, save_csv=False, csv_filename="nnps_results.csv"):
    assert image.shape[0] >= large_roi_size and image.shape[1] >= large_roi_size, "Imagen demasiado pequeña"

    center_y, center_x = np.array(image.shape) // 2
    half = large_roi_size // 2
    region = image[center_y-half:center_y+half, center_x-half:center_x+half]

    region_flat = flatten_image(region)

    step = roi_size // 2
    rois = []
    for y in range(0, large_roi_size - roi_size + 1, step):
        for x in range(0, large_roi_size - roi_size + 1, step):
            roi = region_flat[y:y+roi_size, x:x+roi_size]
            rois.append(roi)

    nps_rois = []
    for roi in rois:
        f = fft2(roi)
        fshift = fftshift(f)
        psd2d = (np.abs(fshift)**2) * (pixel_spacing_mm**2) / (roi_size * roi_size)
        nps_rois.append(psd2d)

    nps_mean = np.mean(nps_rois, axis=0)

    freqs = np.fft.fftfreq(roi_size, d=pixel_spacing_mm)
    freqs = fftshift(freqs)

    mean_signal = np.mean(region)
    nnps = nps_mean / (mean_signal ** 2)

    center = roi_size // 2
    nnps_horizontal = nnps[center, :]
    nnps_vertical = nnps[:, center]

    Y, X = np.indices(nnps.shape)
    R = np.sqrt((X - center)**2 + (Y - center)**2)
    R = R.astype(np.int32)
    radial_prof = np.bincount(R.ravel(), nnps.ravel()) / np.bincount(R.ravel())

    if save_csv:
        mask = freqs >= 0
        freqs_pos = freqs[mask]
        nnps_horizontal_pos = nnps_horizontal[mask]
        nnps_vertical_pos = nnps_vertical[mask]
        radial_prof_pos = radial_prof[:len(freqs_pos)]

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frecuencia (ciclos/mm)', 'NNPS_Horizontal', 'NNPS_Vertical', 'NNPS_Radial'])
            for f, h, v, r in zip(freqs_pos, nnps_horizontal_pos, nnps_vertical_pos, radial_prof_pos):
                writer.writerow([f, h, v, r])

    mask = freqs >= 0
    freqs_pos = freqs[mask]
    nnps_horizontal_pos = nnps_horizontal[mask]
    nnps_vertical_pos = nnps_vertical[mask]
    radial_prof_pos = radial_prof[:len(freqs_pos)]
    return nnps, freqs, nnps_horizontal, nnps_vertical, radial_prof
