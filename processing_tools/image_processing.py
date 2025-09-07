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

def segment_and_draw_rois(image):
    attenuator_rois = detect_attenuators(image)
    if len(attenuator_rois) < 2:
        return image, {}

    roi_small = segment_small_roi(image, attenuator_rois) #atenuador rois ordena los rois atenuadores del más chico al más grande
    roi_main = attenuator_rois[-1]
    roi_background = define_background_roi(roi_main, image.shape)
    roi_h, roi_v = generate_centered_mtf_rois(roi_main, image.shape, size_v=(300, 500), size_h=(500, 300))

    roi_dict = {
        "attenuator_rois": roi_main, #roi de cobre
        "background_roi": roi_background,
        "roi_horizontal": roi_h,
        "roi_vertical": roi_v,
        "roi_small": roi_small #roi de aluminio
    }

    all_rois = [roi_main, roi_background, roi_h, roi_v, roi_small]
    colors = [(255, 0, 255), (0, 255, 255), (0, 255, 0), (0, 0, 255), (0, 255, 0)]
    drawn = draw_rois(image, all_rois, colors)

    return drawn, roi_dict