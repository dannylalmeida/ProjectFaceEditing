from __future__ import annotations

import unicodedata

from src.legacy import ensure_legacy_scripts_on_path


def get_region_labels() -> dict[str, list[int]]:
    ensure_legacy_scripts_on_path()
    from face_pipeline_utils import REGION_LABELS

    return REGION_LABELS


EDIT_REGION_ALIASES = {
    "auto": "auto",
    "mouth": "mouth",
    "boca": "mouth",
    "smile": "mouth",
    "sorriso": "mouth",
    "lips": "mouth",
    "lip": "mouth",
    "labios": "mouth",
    "labio": "mouth",
    "face": "face",
    "pele": "face",
    "skin": "face",
    "age": "face",
    "idade": "face",
    "older": "face",
    "younger": "face",
    "eyes": "eyes",
    "eye": "eyes",
    "olhos": "eyes",
    "olho": "eyes",
    "iris": "iris",
    "irises": "iris",
    "eyebrows": "eyebrows",
    "eyebrow": "eyebrows",
    "brows": "eyebrows",
    "sobrancelhas": "eyebrows",
    "sobrancelha": "eyebrows",
    "nose": "nose",
    "nariz": "nose",
    "ears": "ears",
    "ear": "ears",
    "orelhas": "ears",
    "orelha": "ears",
    "neck": "neck",
    "pescoco": "neck",
}


EDIT_REGION_MASK_DEFAULTS = {
    "mouth": {"dilation": 1, "blur": 4},
    "face": {"dilation": 4, "blur": 9},
    "eyes": {"dilation": 0, "blur": 2},
    "iris": {"dilation": 0, "blur": 1},
    "eyebrows": {"dilation": 1, "blur": 3},
    "nose": {"dilation": 0, "blur": 0},
    "ears": {"dilation": 1, "blur": 4},
    "neck": {"dilation": 2, "blur": 6},
}


def normalize_edit_region(edit_region: str | None) -> str:
    raw = (edit_region or "auto").strip().lower()
    decomposed = unicodedata.normalize("NFKD", raw)
    normalized = "".join(char for char in decomposed if not unicodedata.combining(char))
    return EDIT_REGION_ALIASES.get(normalized, normalized)


def build_region_mask(parsing_map, labels: list[int], dilation: int, cv2, np):
    mask = np.isin(parsing_map, labels).astype(np.uint8) * 255
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def create_edit_mask(attribute: str, parsing_map, cv2, np, dilation: int = 0):
    labels_by_region = get_region_labels()
    attribute = normalize_edit_region(attribute)

    if attribute == "mouth":
        labels = labels_by_region["boca"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "face":
        labels = labels_by_region["pele"] + labels_by_region["nariz"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute in {"eyes", "iris"}:
        labels = labels_by_region["olhos"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "eyebrows":
        labels = labels_by_region["sobrancelhas"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "nose":
        labels = labels_by_region["nariz"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "ears":
        labels = labels_by_region["orelhas"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "neck":
        labels = labels_by_region["pescoco"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    labels = labels_by_region["pele"]
    return build_region_mask(parsing_map, labels, dilation, cv2, np), labels


def get_edit_mask_defaults(attribute: str) -> dict[str, int]:
    attribute = normalize_edit_region(attribute)
    return EDIT_REGION_MASK_DEFAULTS.get(attribute, EDIT_REGION_MASK_DEFAULTS["face"])


def refine_mask(mask_uint8, dilation: int, erosion: int, blur: int, threshold: int, cv2, np):
    refined = (mask_uint8 > max(0, min(threshold, 255))).astype(np.uint8) * 255
    if erosion > 0:
        kernel_size = erosion * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        refined = cv2.erode(refined, kernel, iterations=1)
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        refined = cv2.dilate(refined, kernel, iterations=1)
    if blur > 0:
        blur_size = blur * 2 + 1
        refined = cv2.GaussianBlur(refined, (blur_size, blur_size), 0)
    return np.clip(refined, 0, 255).astype(np.uint8)


def build_soft_mask(mask_uint8, blur: int, cv2, np):
    if blur > 0:
        blur_size = blur * 2 + 1
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 0)
    return np.clip(mask_uint8.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]


def extract_mask_outline_contours(
    mask_uint8,
    cv2,
    np,
    simplify_epsilon_ratio: float = 0.004,
    min_area: float = 4.0,
) -> list[list[list[int]]]:
    hard = (mask_uint8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines: list[list[list[int]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        epsilon = max(1.0, cv2.arcLength(contour, True) * simplify_epsilon_ratio)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(point[0][0]), int(point[0][1])] for point in simplified]
        if len(points) >= 3:
            outlines.append(points)
    return outlines


def build_mask_overlay(base_bgr, mask_uint8, cv2, np, color_bgr=(0, 255, 0), alpha: float = 0.24):
    color = np.zeros_like(base_bgr)
    color[..., 0] = color_bgr[0]
    color[..., 1] = color_bgr[1]
    color[..., 2] = color_bgr[2]
    soft = build_soft_mask(mask_uint8, 2, cv2, np) * float(max(0.0, min(alpha, 1.0)))
    overlay = (
        color.astype(np.float32) * soft
        + base_bgr.astype(np.float32) * (1.0 - soft)
    ).clip(0, 255).astype(np.uint8)
    hard = (mask_uint8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        thickness = max(2, int(round(min(base_bgr.shape[:2]) * 0.004)))
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.drawContours(overlay, contours, -1, color_bgr, thickness, cv2.LINE_AA)
    return overlay
