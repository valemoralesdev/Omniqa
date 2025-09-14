import cv2
import numpy as np

def preprocess_image(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def detect_attenuators(image):
    processed_image = preprocess_image(image)
    _, thresh = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    min_size = 20
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_size and h > min_size:
            rois.append((x, y, w, h))

    return sorted(rois, key=lambda r: r[2] * r[3])

def segment_small_roi(image, all_rois):
    sorted_rois = sorted(all_rois, key=lambda r: r[2] * r[3])
    for roi in sorted_rois:
        x, y, w, h = roi
        if w > 10 and h > 10:
            new_w = int(w * 0.6)
            new_h = int(h * 0.6)
            new_x = x + (w - new_w) // 2
            new_y = y + (h - new_h) // 2
            return (new_x, new_y, new_w, new_h)
    return None

def define_background_roi(attenuator_roi, image_shape):
    x, y, w, h = attenuator_roi
    roi_w = int(w * 1.2)
    roi_h = int(h * 1.2)
    roi_x = min(image_shape[1] - roi_w - 10, x + int(w * 2.8))  # más a la derecha
    roi_y = max(0, y - int(h * 0.2))
    return (roi_x, roi_y, roi_w, roi_h)

def generate_mtf_rois(main_roi):
    x, y, w, h = main_roi
    roi_v_w = int(w * 0.6)
    roi_v = (x + w // 2 - roi_v_w // 2, y + int(h * 0.6), roi_v_w, h)

    roi_h_h = int(h * 0.6)
    roi_h = (x + int(w * 0.6), y + h // 2 - roi_h_h // 2, w, roi_h_h)
    return roi_h, roi_v

def draw_rois(image, rois, colors):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for roi, color in zip(rois, colors):
        x, y, w, h = roi
        cv2.rectangle(image_color, (x, y), (x + w, y + h), color, 2)
    return image_color

def generate_centered_mtf_rois(main_roi, image_shape, size_v=(80, 160), size_h=(160, 80)):
    """
    Genera ROIs para cálculo de MTF estilo IAEA:
    - Centrados sobre el borde inferior (vertical) y derecho (horizontal) del atenuador.
    - Mitad adentro / mitad afuera.
    - Tamaños fijos configurables.
    """
    x, y, w, h = main_roi
    img_h, img_w = image_shape

    # ROI vertical: centrado sobre borde inferior
    rv_w, rv_h = size_v
    rv_x = x + w // 2 - rv_w // 2
    rv_y = y + h - rv_h // 2
    roi_v = (max(0, rv_x), max(0, rv_y), rv_w, rv_h)

    # ROI horizontal: centrado sobre borde derecho
    rh_w, rh_h = size_h
    rh_x = x + w - rh_w // 2
    rh_y = y + h // 2 - rh_h // 2
    roi_h = (max(0, rh_x), max(0, rh_y), rh_w, rh_h)

    return roi_h, roi_v

def segment_and_draw_rois(image, pixel_spacing_mm):
    """
    Genera ROIs estilo IAEA (40x40 mm) y los re-centra para que el borde
    quede exactamente en el centro (half-in/half-out) usando la recta del borde.
    """
    attenuator_rois = detect_attenuators(image)
    if len(attenuator_rois) < 2:
        return image, {}, {}

    roi_small = segment_small_roi(image, attenuator_rois)  # aluminio (el más chico)
    roi_main  = attenuator_rois[-1]                        # cobre (el más grande)
    roi_background = define_background_roi(roi_main, image.shape)

    # ROIs 40x40 mm sobre bordes derecho (H) e inferior (V)
    roi_h, roi_v = generate_centered_mtf_rois_mm(
        roi_main, image.shape, pixel_spacing_mm, size_mm=(40.0, 40.0)
    )

    # === Re-centrado geométrico por recta del borde (mejor que gradiente) ===
    roi_h, ang_h = recenter_roi_to_edge(image, roi_h, max_shift_px=6)
    roi_v, ang_v = recenter_roi_to_edge(image, roi_v, max_shift_px=6)
    angles = {"roi_horizontal": ang_h, "roi_vertical": ang_v}
    # =======================================================================

    roi_dict = {
        "attenuator_rois": roi_main,
        "background_roi": roi_background,
        "roi_horizontal": roi_h,  # borde derecho (evalúa MTF horizontal)
        "roi_vertical": roi_v,    # borde inferior (evalúa MTF vertical)
        "roi_small": roi_small
    }

    # Dibujar AFTER re-centrar
    all_rois = [roi_main, roi_background, roi_h, roi_v, roi_small]
    colors   = [(255, 0, 255), (0, 255, 255), (0, 255, 0), (0, 0, 255), (0, 255, 0)]
    drawn = draw_rois(image, all_rois, colors)

    return drawn, roi_dict, angles


# --- Ya lo tenés, lo dejo aquí por claridad ---
def generate_centered_mtf_rois_mm(main_roi, image_shape, pixel_spacing_mm, size_mm=(40.0, 40.0)):
    """
    ROIs estilo IAEA:
    - 40x40 mm por defecto (size_mm).
    - Centrados sobre los bordes 'útiles' del atenuador: derecho (horizontal) e inferior (vertical).
    - Mitad adentro / mitad afuera del borde.
    """
    x, y, w, h = main_roi
    H, W = image_shape[:2]

    w_px = int(round(size_mm[0] / pixel_spacing_mm))
    h_px = int(round(size_mm[1] / pixel_spacing_mm))

    # ROI vertical (borde inferior)
    rv_x = x + w // 2 - w_px // 2
    rv_y = y + h - h_px // 2
    roi_v = (max(0, rv_x), max(0, rv_y), w_px, h_px)

    # ROI horizontal (borde derecho)
    rh_x = x + w - w_px // 2
    rh_y = y + h // 2 - h_px // 2
    roi_h = (max(0, rh_x), max(0, rh_y), w_px, h_px)

    return roi_h, roi_v


import numpy as np
import cv2

def recenter_roi_to_edge(image, roi, max_shift_px=6):
    """
    Re-centra el ROI para que la recta del borde pase por el centro del ROI.
    - Detecta borde por Canny dentro del ROI, ajusta y = m x + b.
    - Calcula la distancia ortogonal del centro del ROI a la recta.
    - Desplaza el ROI a lo largo de la normal hasta anular esa distancia.
    """
    x, y, w, h = roi
    sub = image[y:y+h, x:x+w].astype(np.float32)

    # Edge detection robusto en 8 bits (mejor Canny)
    u8 = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(u8, 40, 120)

    ys, xs = np.where(edges > 0)
    if xs.size < 30:
        # Sin bordes detectables: no toco el ROI
        return roi, None

    # Ajuste de recta y = m x + b en coords locales del ROI
    m, b = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 1)
    angle_deg = float(np.degrees(np.arctan(m)))

    # Distancia ortogonal (con signo) del centro del ROI a la recta
    cx, cy = w / 2.0, h / 2.0
    denom = np.sqrt(m * m + 1.0)
    d = (-m * cx + cy - b) / denom  # en píxeles

    # Vector normal unitario a la recta (en coords de imagen local)
    nx, ny = -m / denom, 1.0 / denom

    # Desplazamiento requerido para que el borde pase por el centro
    shift_x = -d * nx
    shift_y = -d * ny

    # Limitar y redondear a píxeles enteros
    shift_x = int(np.clip(np.round(shift_x), -max_shift_px, max_shift_px))
    shift_y = int(np.clip(np.round(shift_y), -max_shift_px, max_shift_px))

    # Aplicar desplazamiento en coords de imagen global, con clamping a bordes
    xx = int(np.clip(x + shift_x, 0, image.shape[1] - w))
    yy = int(np.clip(y + shift_y, 0, image.shape[0] - h))

    return (xx, yy, w, h), angle_deg

