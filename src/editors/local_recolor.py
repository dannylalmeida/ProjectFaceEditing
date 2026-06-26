from __future__ import annotations

import re

from src.segmentation.mask_utils import build_soft_mask


COLOR_NAME_MAP = {
    "green": "green",
    "verde": "green",
    "blue": "blue",
    "azul": "blue",
    "azuis": "blue",
    "brown": "brown",
    "castanho": "brown",
    "castanha": "brown",
    "castanhos": "brown",
    "castanhas": "brown",
    "hazel": "hazel",
    "gray": "gray",
    "grey": "gray",
    "cinzento": "gray",
    "cinzenta": "gray",
    "cinza": "gray",
    "black": "black",
    "preto": "black",
    "preta": "black",
    "white": "white",
    "branco": "white",
    "branca": "white",
    "red": "red",
    "vermelho": "red",
    "vermelha": "red",
    "pink": "pink",
    "rosa": "pink",
    "purple": "purple",
    "roxo": "purple",
    "roxa": "purple",
    "blond": "blond",
    "blonde": "blond",
    "loiro": "blond",
    "loira": "blond",
    "gold": "gold",
    "golden": "gold",
    "dourado": "gold",
    "dourada": "gold",
}


COLOR_BGR_MAP = {
    "green": (70, 135, 55),
    "blue": (230, 95, 20),
    "brown": (70, 95, 130),
    "hazel": (70, 120, 105),
    "gray": (130, 135, 140),
    "black": (35, 35, 35),
    "white": (220, 220, 220),
    "red": (60, 60, 190),
    "pink": (155, 120, 210),
    "purple": (150, 90, 155),
    "blond": (125, 185, 225),
    "gold": (90, 170, 220),
}


def infer_target_color(description: str) -> str | None:
    normalized = (description or "").strip().lower()
    for color_word, canonical in COLOR_NAME_MAP.items():
        if re.search(rf"(?<!\w){re.escape(color_word)}(?!\w)", normalized):
            return canonical
    return None


def recolor_iris(base_bgr, iris_mask_uint8, color_name: str, strength: float, cv2, np, blur: int = 1):
    target_bgr = COLOR_BGR_MAP.get(color_name)
    if target_bgr is None or not np.any(iris_mask_uint8):
        return base_bgr

    alpha = build_soft_mask(iris_mask_uint8, blur, cv2, np)
    alpha *= float(max(0.0, min(strength, 1.0)))
    alpha_2d = alpha[..., 0]

    hsv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(np.uint8([[list(target_bgr)]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)

    desired = hsv.copy()
    desired[..., 0] = target_hsv[0]
    desired[..., 1] = np.maximum(hsv[..., 1] * 0.50 + target_hsv[1] * 0.95, 175.0)
    desired[..., 2] = np.clip(hsv[..., 2] * 1.12 + 14.0, 0, 255)

    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pupil_or_lash = ((gray < 38) & (iris_mask_uint8 > 0)).astype(np.float32)
    highlight = ((gray > 208) & (iris_mask_uint8 > 0)).astype(np.float32)
    preserve = np.clip(pupil_or_lash + highlight, 0.0, 1.0)
    alpha_2d = alpha_2d * (1.0 - preserve * 0.90)

    desired_bgr = cv2.cvtColor(desired.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    result = (
        desired_bgr * alpha_2d[..., None]
        + base_bgr.astype(np.float32) * (1.0 - alpha_2d[..., None])
    )
    return result.clip(0, 255).astype(np.uint8)

