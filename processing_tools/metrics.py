# ========================= metrics.py =========================
import numpy as np
import cv2
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import windows
from scipy.fft import fft, fftfreq

# ----------------------- Métricas básicas ----------------------

def _roi_slice(roi):
    x, y, w, h = roi
    return slice(y, y + h), slice(x, x + w)

def calculate_snr(image_normalized, background_roi):
    """
    SNR en fondo: mean_bg / std_bg
    Espera 'image_normalized' (p.ej. imagen/mean_bg) o la imagen cruda;
    en ambos casos hace mean y std sobre el ROI de fondo.
    """
    ys, xs = _roi_slice(background_roi)
    bg = image_normalized[ys, xs].astype(np.float64)
    mu = float(np.mean(bg))
    sigma = float(np.std(bg, ddof=1) if bg.size > 1 else 0.0)
    return mu / sigma if sigma > 0 else np.nan

def calculate_sdnr(image_normalized, signal_roi, background_roi):
    """
    SDNR = (mean_signal - mean_bg) / std_bg
    """
    ys, xs = _roi_slice(signal_roi)
    sig = image_normalized[ys, xs].astype(np.float64)
    yb, xb = _roi_slice(background_roi)
    bg = image_normalized[yb, xb].astype(np.float64)
    mu_s = float(np.mean(sig))
    mu_b = float(np.mean(bg))
    sigma_b = float(np.std(bg, ddof=1) if bg.size > 1 else 0.0)
    return (mu_s - mu_b) / sigma_b if sigma_b > 0 else np.nan

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d, uniform_filter1d
from scipy.signal import windows
from scipy.fft import fft, fftfreq

def pmft_horizontal_iaea_like(
    image,
    roi_horizontal_edge,                 # ROI del borde HORIZONTAL (borde INFERIOR del cobre)
    pixel_spacing_mm=0.1,                # pitch (mm/píxel)
    angle_tilt_deg=-2.43,                # tilt que ves en IAEA para el borde horizontal
    super_sampling_factor=10,            # 1/10 de píxel
    hann_window_mm=25.0,                 # ventana sobre LSF
    bin_pitch=0.25,                      # rejilla de salida (mm^-1)
    out_png="images/pMTF_horizontal_debug.png",
    smooth_esf="median5",                # "median5" (IAEA) o "gauss0.4" (opcional)
    smooth_pmtf=False,                   # si querés, suavizado leve de pMTF (no recomendado)
):
    """
    Replica el flujo IAEA para pMTF SOLO del borde horizontal (inferior):
      - Proyección ortogonal y supermuestreo 1/10 px (ESF)
      - Centrado por máximo gradiente y recorte a ±12.5 mm (≈25 mm)
      - Normalización por colas (15%) y suavizado (mediana 5)
      - LSF = derivada; quitar DC por colas (5%)
      - Ventana Hann 25 mm centrada en el medio geométrico
      - FFT -> pMTF; normalizar a 1 en f=0; rebin a bin_pitch
      - Cálculo de pMTF50 / pMTF20 / pMTF10 y pMTF@Nyquist
      - Figura de 4 paneles estilo IAEA
    Devuelve: (f_grid, pmtf, metrics, debug)
    """

    # --------------------------
    # 0) ROI y recta del borde
    x, y, w, h = roi_horizontal_edge
    roi = image[y:y+h, x:x+w].astype(np.float64)

    # Borde horizontal ideal = 0°, aplicamos tilt
    angle_raw_deg = (0.0 + angle_tilt_deg) % 180.0
    m = 0.0 if abs(angle_raw_deg) < 1e-12 else np.tan(np.radians(angle_raw_deg))
    cx, cy = (w/2.0), (h/2.0)
    b = cy - m*cx
    denom = np.sqrt(m*m + 1.0)

    # --------------------------
    # 1) ESF por proyección ortogonal (1/10 de píxel)
    dx = pixel_spacing_mm / float(super_sampling_factor)   # mm por muestra ESF/LSF

    # Rango geométrico (dist a la recta de las 4 esquinas) + padding
    corners = np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]], dtype=np.float64)
    d_corners = (-m*corners[:,0] + corners[:,1] - b) / denom
    dmin, dmax = float(np.min(d_corners)), float(np.max(d_corners))
    dmin -= 2.0; dmax += 2.0

    pos_min = dmin * super_sampling_factor
    pos_max = dmax * super_sampling_factor
    nbins   = int(np.ceil(pos_max - pos_min)) + 3
    nbins   = max(nbins, 64)

    esf = np.zeros(nbins, np.float64)
    cnt = np.zeros(nbins, np.float64)
    base = -pos_min

    jj = np.arange(h, dtype=np.float64)
    for j in range(h):
        ii   = np.arange(w, dtype=np.float64)
        pos  = ((-m*ii + jj[j] - b) / denom) * super_sampling_factor + base
        i0   = np.floor(pos).astype(int)
        frac = pos - i0
        v    = roi[j, :]

        valid0 = (i0 >= 0) & (i0 < nbins)
        valid1 = (i0+1 >= 0) & (i0+1 < nbins)

        np.add.at(esf, i0[valid0], (1.0 - frac[valid0]) * v[valid0])
        np.add.at(cnt, i0[valid0], (1.0 - frac[valid0]))
        np.add.at(esf, (i0+1)[valid1], frac[valid1] * v[valid1])
        np.add.at(cnt, (i0+1)[valid1], frac[valid1])

    good = cnt > 1e-6
    i0 = int(np.argmax(good))
    i1 = int(len(good) - np.argmax(good[::-1]) - 1)
    esf = (esf[i0:i1+1] / np.maximum(cnt[i0:i1+1], 1e-9)).astype(np.float64)

    # --------------------------
    # 2) Centrado por máximo gradiente y recorte a ±12.5 mm (≈25 mm)
    def _center_by_grad(esf_vec, dx_mm):
        if esf_vec.size < 9: return esf_vec.size // 2
        esf_s = gaussian_filter1d(esf_vec, sigma=1.0)
        g = np.abs(np.gradient(esf_s))
        return int(np.argmax(g))

    c = _center_by_grad(esf, dx)
    half = int(round(12.5 / dx))
    c = max(half, min(len(esf)-1-half, c))
    esf = esf[c-half: c+half+1]  # ≈ 25 mm

    # Para eje x (mm) en las figuras
    x_mm = (np.arange(len(esf)) - len(esf)//2) * dx

    # --------------------------
    # 3) Normalización por colas (15%) y suavizado ESF
    k = max(10, int(0.15 * len(esf)))
    low_med  = float(np.median(esf[:k]))
    high_med = float(np.median(esf[-k:]))
    step = high_med - low_med
    if step < 0:
        esf = -esf
        low_med, high_med = -high_med, -low_med
        step = -step
    if step < 1e-6:
        raise RuntimeError("Paso de ESF ~0; revisar ROI horizontal.")

    # ¡Sin clip para el cálculo!
    esf01_proc = (esf - low_med) / step

    # Suavizado estilo IAEA
    if smooth_esf == "median5":
        esf_s = median_filter(esf01_proc, size=5, mode="nearest")
    elif smooth_esf == "gauss0.4":
        esf_s = gaussian_filter1d(esf01_proc, sigma=0.4)
    else:
        esf_s = esf01_proc.copy()

    # (Opcional solo para PLOT)
    esf01_plot = np.clip(esf01_proc, -0.2, 1.2)
    esf_s_plot = np.clip(esf_s,      -0.2, 1.2)

    # --------------------------
    # 4) LSF = derivada; quitar DC por colas (5%); Hann centrada
    lsf = np.gradient(esf_s).astype(np.float64)
    n = len(lsf)
    tails = max(5, int(0.05 * n))
    dc = 0.5 * (float(np.mean(lsf[:tails])) + float(np.mean(lsf[-tails:])))
    lsf -= dc

    win_len = int(round(hann_window_mm / dx))
    win_len = max(33, min(win_len, n))
    k_center = n // 2
    half_w = min(win_len // 2, k_center, n - 1 - k_center)
    win = np.zeros(n, np.float64)
    win[k_center - half_w : k_center + half_w + 1] = windows.hann(2*half_w + 1, sym=True)

    lsf_win = lsf * win
    # Normalización de área DESPUÉS del Hann
    area = float(np.trapz(lsf_win, dx=dx))
    if abs(area) > 1e-9:
        lsf_win /= area
    # (remache opcional a suma constante)
    suma = np.sum(lsf_win)
    if abs(suma) > 1e-12:
        lsf_win /= suma

    # --------------------------
    # 5) FFT -> pMTF y rebin
    F = np.abs(fft(lsf_win))
    f = fftfreq(n, d=dx)            # lp/mm
    sel = f >= 0
    f = f[sel]
    pmtf = F[sel]

    # Normalizar pMTF a 1 en f=0 ANTES de cualquier suavizado
    pmtf = np.clip(pmtf / max(pmtf[0], 1e-12), 0.0, 1.0)

    # Rejilla de salida
    if bin_pitch and bin_pitch > 0:
        f_grid = np.arange(0.0, float(f[-1]) + 0.5*bin_pitch, bin_pitch, dtype=float)
        pmtf_r = np.interp(f_grid, f, pmtf)
    else:
        f_grid, pmtf_r = f.copy(), pmtf.copy()

    # Suavizado de pMTF (opcional y leve; por defecto OFF para igualar IAEA)
    if smooth_pmtf and pmtf_r.size >= 3:
        pmtf_r = uniform_filter1d(pmtf_r, size=3, mode="nearest")
    # NO re-normalizar pMTF aquí
    pmtf_r = np.clip(pmtf_r, 0.0, 1.0)

    # --------------------------
    # 6) Métricas
    def _crossing(x, y, level):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        idx = np.where((y[:-1]-level)*(y[1:]-level) <= 0)[0]
        if idx.size == 0: return float("nan")
        i = int(idx[0]); x0,x1,y0,y1 = x[i],x[i+1],y[i],y[i+1]
        if y1 == y0: return float("nan")
        return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))

    f_nyq = 1.0 / (2.0 * pixel_spacing_mm)
    p_at_nyq = float(np.interp(f_nyq, f_grid, pmtf_r))

    mask = (f_grid <= 15.0)
    fx, py = f_grid[mask], pmtf_r[mask]
    mtf50 = _crossing(fx, py, 0.5)
    mtf20 = _crossing(fx, py, 0.2)
    mtf10 = _crossing(fx, py, 0.1)

    metrics = {
        "Nyquist_freq_mm^-1": f_nyq,
        "pMTF@Nyquist": p_at_nyq,
        "MTF@50%": mtf50,
        "MTF@20%": mtf20,
        "MTF@10%": mtf10,
        "angle_deg_raw": angle_raw_deg,
        "angle_deg_tilt": angle_tilt_deg,
    }

    # Tabla estilo “Results” (0…15 mm^-1 cada 0.25)
    table_f = np.arange(0.0, 15.25+1e-9, 0.25)
    table_m = np.interp(table_f, f_grid, pmtf_r)

    # --------------------------
    # 7) Figura 4 paneles estilo IAEA
    fig, ax = plt.subplots(2,2, figsize=(11,7), dpi=120)

    ax[0,0].plot(x_mm, esf, lw=1)
    ax[0,0].set_title("Original oversampled edge")
    ax[0,0].set_xlabel("mm"); ax[0,0].set_ylabel("ESF (a.u.)")

    ax[0,1].plot(x_mm, esf_s_plot, lw=1)
    ttl = "Smoothed (median filter on 5 mobile points)" if smooth_esf=="median5" else "Smoothed ESF (Gaussian)"
    ax[0,1].set_title(ttl)
    ax[0,1].set_xlabel("mm"); ax[0,1].set_ylabel("ESF (a.u.)")

    ax[1,0].plot(x_mm, lsf_win, lw=1)
    ax[1,0].set_title("Line Spread Function (central portion)")
    ax[1,0].set_xlabel("mm"); ax[1,0].set_ylabel("LSF (a.u.)")

    ax[1,1].plot(f_grid, pmtf_r, lw=1, label="pMTF")
    ax[1,1].set_xlim(0, 15.5); ax[1,1].set_ylim(0, 1.02)
    ax[1,1].set_title("pMTF - rebinned to 0.25 mm$^{-1}$")
    ax[1,1].set_xlabel("spatial frequency [mm$^{-1}$]"); ax[1,1].set_ylabel("pMTF")
    ax[1,1].legend(loc="upper right")

    # Popup/resumen
    txt = (f"Nyquist Freq (mm$^{{-1}}$): {f_nyq:.2f}\n"
           f"pMTF$_{{50}}$ is {mtf50:.2f}\n"
           f"pMTF$_{{10}}$ is {mtf10:.2f}\n"
           f"pMTF (%) at Nyquist is {100*p_at_nyq:.1f}")
    ax[1,1].text(0.98, 0.35, txt, transform=ax[1,1].transAxes,
                 ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))

    fig.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"🟢 Figura guardada en {out_png}")
    print(f"Edge angle is: {angle_tilt_deg:+.2f} °")
    print(f"Nyquist={f_nyq:.2f}, pMTF50={mtf50:.2f}, pMTF10={mtf10:.2f}, pMTF@Nyq={100*p_at_nyq:.1f}%")

    debug = {
        "x_mm": x_mm,
        "ESF_raw": esf,              # ESF recortada a ±12.5 mm (sin normalizar)
        "ESF_norm": esf01_proc,      # ESF normalizada (sin clip)
        "ESF_smooth": esf_s,         # ESF normalizada y suavizada
        "LSF_windowed": lsf_win,
        "f_grid": f_grid,
        "pMTF": pmtf_r,
        "table_freqs": table_f,
        "table_pMTF": table_m,
        "area_after_hann": area,
        "sum_after_hann": float(np.sum(lsf_win)),
    }
    return f_grid, pmtf_r, metrics, debug

