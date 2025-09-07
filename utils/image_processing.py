import cv2
import numpy as np

def preprocess_image(image):
    """
    Aplica normalización y filtro gaussiano para mejorar la segmentación.
    """
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def detect_attenuators(image):
    """
    Detecta los elementos atenuantes en la imagen.
    """
    processed_image = preprocess_image(image)
    _, thresh = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    min_size = 20  # Tamaño mínimo para filtrar ruido
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_size and h > min_size:
            rois.append((x, y, w, h))
    
    return rois

def define_background_roi(rois, image_shape):
    """
    Define la ROI de fondo basada en la posición de los otros elementos.
    """
    image_h, image_w = image_shape[:2]
    background_x = int(0.75 * image_w)
    background_y = int(0.2 * image_h)
    background_w = int(0.2 * image_w)
    background_h = int(0.2 * image_h)
    
    return (background_x, background_y, background_w, background_h)

def generate_perpendicular_rois(main_roi, image_shape):
    """
    Genera dos rectángulos perpendiculares que comparten el centro con el ROI más grande.
    Uno es horizontal y el otro vertical, con desplazamiento.
    """
    x, y, w, h = main_roi
    center_x, center_y = x + w // 2, y + h // 2  # Centro del ROI
    max_side = max(w, h) // 2  # Mitad del lado más largo

    # Rectángulo horizontal (mismo ancho, mitad de altura), desplazado a la derecha
    horizontal_roi = (
        min(image_shape[1] - w, x + max_side),  # Desplazado a la derecha
        center_y - (h // 4),
        w, h // 2
    )

    # Rectángulo vertical (mismo alto, mitad de ancho), desplazado hacia arriba
    vertical_roi = (
        center_x - (w // 4),
        max(0, y - max_side),  # Desplazado hacia arriba
        w // 2, h
    )

    return horizontal_roi, vertical_roi

def draw_rois(image, rois, colors):
    """
    Dibuja los ROIs en la imagen con los colores especificados.
    """
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for roi, color in zip(rois, colors):
        x, y, w, h = roi
        cv2.rectangle(image_color, (x, y), (x + w, y + h), color, 2)
    return image_color

def segment_and_draw_rois(image):
    """
    Detecta y segmenta los ROIs en la imagen, luego los dibuja.
    """
    attenuator_rois = detect_attenuators(image)
    
    if not attenuator_rois:
        return image, []  # Si no hay elementos detectados, devuelve la imagen original y una lista vacía

    background_roi = define_background_roi(attenuator_rois, image.shape)
    largest_attenuator_roi = max(attenuator_rois, key=lambda r: r[2] * r[3])
    horizontal_roi, vertical_roi = generate_perpendicular_rois(largest_attenuator_roi, image.shape)

    all_rois = attenuator_rois + [background_roi, horizontal_roi, vertical_roi]
    colors = [(255, 0, 0)] * len(attenuator_rois) + [(0, 255, 255), (0, 255, 0), (255, 0, 0)]  # Rojo para atenuadores, amarillo para fondo, verde y azul para MTF

    return draw_rois(image, all_rois, colors), all_rois
