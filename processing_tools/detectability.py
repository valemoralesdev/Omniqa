from scipy.interpolate import interp1d
from scipy.special import j1
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
import numpy as np
from processing_tools.nnps import extract_roi

def calculate_dprime(freqs_horizontal, mtf_horizontal, freqs_vertical, mtf_vertical, roi_background, roi_signal=None):
    freqs_common = np.linspace(0.01, 5, 300)
    mtf_horizontal_interp = np.interp(freqs_common, freqs_horizontal, mtf_horizontal)
    mtf_vertical_interp = np.interp(freqs_common, freqs_vertical, mtf_vertical)
    mtf_mean = (mtf_horizontal_interp + mtf_vertical_interp) / 2

    roi_background_512 = extract_roi(roi_background, centered=True, size=(512, 512))
    flat = roi_background_512 - gaussian_filter(roi_background_512, sigma=20)
    win = np.outer(np.hanning(512), np.hanning(512))
    windowed = flat * win
    nps = np.abs(fftshift(fft2(windowed)))**2
    nps /= (512**2 * np.mean(flat)**2)

    def radial_profile(data):
        y, x = np.indices(data.shape)
        r = np.hypot(x - 256, y - 256).astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        return tbin / np.maximum(nr, 1)

    nnps_profile = radial_profile(nps)
    nnps_profile /= nnps_profile[0]  # Normalización

    nyquist = 1 / (2 * 0.1)
    freqs_nnps = np.linspace(0, nyquist, len(nnps_profile))

    def S_u(u, D):
        x = np.pi * D/2 * u
        return np.where(x == 0, 1.0, 2 * j1(x) / x)

    def VTF(u):
        return np.exp(-0.7 * u)

    nnps_interp = interp1d(freqs_nnps, nnps_profile, bounds_error=False, fill_value="extrapolate")
    mtf_interp = interp1d(freqs_common, mtf_mean, bounds_error=False, fill_value=0.0)

    # Calcular contraste real si se proporciona roi_signal
   
    if roi_signal is not None and isinstance(roi_signal, (tuple, list)) and len(roi_signal) == 4:
        x_s, y_s, w_s, h_s = roi_signal
        roi_signal_data = roi_signal[y_s:y_s+h_s, x_s:x_s+w_s] if isinstance(roi_signal, np.ndarray) else None
        if roi_signal_data is None:
            roi_signal_data = extract_roi(roi_background[y_s:y_s+h_s, x_s:x_s+w_s], centered=True, size=(32, 32))
        else:
            roi_signal_data = extract_roi(roi_signal_data, centered=True, size=(32, 32))

        mean_signal = np.mean(roi_signal_data)
        mean_background = np.mean(roi_background_512)
        C = np.abs(mean_signal - mean_background) / mean_background if mean_background != 0 else 0.4
    else:
        C = 0.4  # fallback default

        
    results_dprime = {}
    for D in [0.25, 0.1]:
        S = S_u(freqs_common, D)
        M = mtf_interp(freqs_common)
        V = VTF(freqs_common)
        N = nnps_interp(freqs_common)
        num = 2 * np.pi * C * np.trapz((S**2) * (M**2) * (V**2) * freqs_common, freqs_common)
        den = np.trapz((S**2) * (M**2) * (V**4) * N * freqs_common, freqs_common)
        d_val = np.sqrt(num / den) if den > 0 else 0.0

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(freqs_common, M, label='MTF promedio')
        plt.plot(freqs_common, V**4, label='VTF⁴')
        plt.plot(freqs_common, N, label='NNPS')
        plt.yscale('log')
        plt.title(f'Curvas para D = {int(D*1000)} µm')
        plt.xlabel('Frecuencia (ciclos/mm)')
        plt.ylabel('Valor')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"curvas_dprime_{int(D*1000)}um.png")
        
        # Graficar S(u) para verificar forma del objeto
        S_01 = S_u(freqs_common, 0.1)
        S_025 = S_u(freqs_common, 0.25)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(freqs_common, S_01**2, label='S² para D = 0.1 mm')
        plt.plot(freqs_common, S_025**2, label='S² para D = 0.25 mm')
        plt.title('Forma espectral del objeto S²(u)')
        plt.xlabel('Frecuencia espacial (ciclos/mm)')
        plt.ylabel('S²(u)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("S_squared_vs_freq.png")

        results_dprime[f"d_prime_{int(D*1000)}um"] = round(float(d_val), 6)

    return results_dprime
        
