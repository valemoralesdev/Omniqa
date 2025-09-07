import numpy as np
import cv2
from processing_tools.metrics import calculate_mtf_npwe


def _synthetic_edge(size=100, angle=5, orientation="vertical"):
    img = np.zeros((size, size), dtype=float)
    if orientation == "vertical":
        img[:, size // 2 :] = 1.0
    else:
        img[size // 2 :, :] = 1.0
    if angle != 0:
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR)
    return img


def test_calculate_mtf_npwe_vertical():
    roi = _synthetic_edge(angle=5, orientation="vertical")
    freqs, mtf, vals = calculate_mtf_npwe(roi, pixel_pitch=0.1, orientation="vertical")
    assert freqs.ndim == 1 and mtf.ndim == 1
    assert len(freqs) == len(mtf)
    assert vals["MTF@50%"] > 0


def test_calculate_mtf_npwe_horizontal():
    roi = _synthetic_edge(angle=5, orientation="horizontal")
    freqs, mtf, vals = calculate_mtf_npwe(roi, pixel_pitch=0.1, orientation="horizontal")
    assert freqs.ndim == 1 and mtf.ndim == 1
    assert len(freqs) == len(mtf)
    assert vals["MTF@20%"] > 0
