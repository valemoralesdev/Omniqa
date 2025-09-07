import numpy as np
import cv2


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *image* by *angle* degrees around its centre using OpenCV."""
    h, w = image.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR)


def _gaussian_smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1‑D Gaussian smoothing using OpenCV kernels."""
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getGaussianKernel(ksize, sigma).flatten()
    return np.convolve(signal, kernel, mode="same")

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

def calcular_mtf_desde_roi(roi_coords, original_image, pixel_spacing=0.1, super_sampling_factor=4):
    from scipy.ndimage import rotate, gaussian_filter1d  # type: ignore
    from scipy.interpolate import interp1d  # type: ignore
    from scipy.fft import fft, fftfreq  # type: ignore
    from sklearn.linear_model import LinearRegression  # type: ignore

    x, y, w, h = roi_coords
    roi = original_image[y:y+h, x:x+w]

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

    angle_correction = angle_deg - 90
    roi_rotated = rotate(roi, angle_correction, reshape=True, order=3)

    crop_w, crop_h = int(roi_rotated.shape[1] * 0.9), int(roi_rotated.shape[0] * 0.9)
    cropped_roi = roi_rotated[roi_rotated.shape[0]//2 - crop_h//2:roi_rotated.shape[0]//2 + crop_h//2,
                              roi_rotated.shape[1]//2 - crop_w//2:roi_rotated.shape[1]//2 + crop_w//2]

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


def calculate_mtf_npwe(roi: np.ndarray, pixel_pitch: float, orientation: str = "vertical"):
    """Calculate MTF using the NPWE edge method.

    Parameters
    ----------
    roi : np.ndarray
        Region of interest containing an edge.
    pixel_pitch : float
        Pixel spacing in millimetres.
    orientation : str, optional
        Orientation of the edge: ``"vertical"`` or ``"horizontal"``.

    Returns
    -------
    freqs : np.ndarray
        Spatial frequency axis (mm^-1).
    mtf : np.ndarray
        Modulation transfer function values.
    mtf_vals : dict
        Frequencies at 50%, 20% and 10% MTF.
    """

    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # --- Half-maximum edge angle estimation ---
    coords = []
    rows, cols = roi.shape
    if orientation == "vertical":
        for y in range(rows):
            line = roi[y, :]
            half = (line.min() + line.max()) / 2.0
            idx = np.where(line >= half)[0]
            if len(idx) == 0 or idx[0] == 0:
                continue
            x1, x2 = idx[0] - 1, idx[0]
            y1, y2 = line[x1], line[x2]
            if y2 == y1:
                continue
            frac = (half - y1) / (y2 - y1)
            coords.append((x1 + frac, y))
        if len(coords) >= 2:
            pts = np.array(coords)
            A = np.vstack([pts[:, 1], np.ones(len(pts))]).T
            m, b = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
            angle = np.degrees(np.arctan(m))
        else:
            angle = 0.0
        roi_rot = _rotate_image(roi, -angle)
        esf = roi_rot.mean(axis=0)
    else:  # horizontal edge
        for x in range(cols):
            line = roi[:, x]
            half = (line.min() + line.max()) / 2.0
            idx = np.where(line >= half)[0]
            if len(idx) == 0 or idx[0] == 0:
                continue
            y1, y2 = idx[0] - 1, idx[0]
            v1, v2 = line[y1], line[y2]
            if v2 == v1:
                continue
            frac = (half - v1) / (v2 - v1)
            coords.append((x, y1 + frac))
        if len(coords) >= 2:
            pts = np.array(coords)
            A = np.vstack([pts[:, 0], np.ones(len(pts))]).T
            m, b = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
            angle = np.degrees(np.arctan(m))
        else:
            angle = 0.0
        roi_rot = _rotate_image(roi, -angle)
        esf = roi_rot.mean(axis=1)

    # --- Super-sampled ESF and smoothing ---
    dx = pixel_pitch / 10.0
    x = np.arange(len(esf)) * pixel_pitch
    x_super = np.arange(x[0], x[-1] + dx, dx)
    esf_super = _gaussian_smooth(np.interp(x_super, x, esf), sigma=1.0)

    # --- LSF via central difference with baseline removal ---
    lsf = np.gradient(esf_super, dx)
    baseline = np.polyval(np.polyfit(x_super, lsf, 1), x_super)
    lsf = lsf - baseline
    lsf /= np.max(np.abs(lsf))

    # --- 12.5 mm Hann window centred on LSF peak ---
    window_mm = 12.5
    window_samples = int(window_mm / dx)
    peak_idx = np.argmax(lsf)
    start = max(0, peak_idx - window_samples // 2)
    end = min(len(lsf), start + window_samples)
    lsf_segment = lsf[start:end]
    window = np.hanning(len(lsf_segment))
    lsf_windowed = lsf_segment * window

    # --- FFT and MTF ---
    freqs = np.fft.fftfreq(len(lsf_windowed), dx)
    pos = freqs >= 0
    freqs = freqs[pos]
    mtf = np.abs(np.fft.fft(lsf_windowed))[pos]
    if mtf[0] != 0:
        mtf /= mtf[0]

    # --- Rebin to 0.05 mm^-1 ---
    freq_step = 0.05
    freq_rebinned = np.arange(0, freqs[-1] + freq_step, freq_step)
    mtf_rebinned = np.interp(freq_rebinned, freqs, mtf, left=0, right=0)

    # --- Interpolate MTF percentages ---
    mtf_vals = {
        "MTF@50%": float(np.round(np.interp(0.5, mtf_rebinned[::-1], freq_rebinned[::-1]), 3)),
        "MTF@20%": float(np.round(np.interp(0.2, mtf_rebinned[::-1], freq_rebinned[::-1]), 3)),
        "MTF@10%": float(np.round(np.interp(0.1, mtf_rebinned[::-1], freq_rebinned[::-1]), 3)),
    }

    return freq_rebinned, mtf_rebinned, mtf_vals
