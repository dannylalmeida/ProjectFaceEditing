from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from face_pipeline_utils import (
    REGION_LABELS,
    build_parsing_model,
    build_region_mask,
    configure_runtime_warnings,
    ensure_directories,
    load_required_modules,
    parse_face_crop,
    suppress_known_stderr_noise,
)
from src.blending.color_match import preserve_original_texture_in_region
from src.editors.local_geometry import resolve_nose_geometry_intent, shrink_masked_region_with_inpaint
from src.segmentation.facemesh_region import build_custom_nose_mask, draw_numbered_facemesh_overlay


REGION_MASK_TUNING = {
    "sobrancelhas": {"dilation_cap": 2, "feather_cap": 5, "diff_threshold": 4, "growth_padding": 8},
    "olhos": {"dilation_cap": 2, "feather_cap": 5, "diff_threshold": 4, "growth_padding": 8},
    "boca": {"dilation_cap": 3, "feather_cap": 7, "diff_threshold": 5, "growth_padding": 10},
    "nariz": {"dilation_cap": 3, "feather_cap": 7, "diff_threshold": 6, "growth_padding": 10},
    "orelhas": {"dilation_cap": 3, "feather_cap": 7, "diff_threshold": 6, "growth_padding": 10},
    "pescoco": {"dilation_cap": 4, "feather_cap": 9, "diff_threshold": 6, "growth_padding": 12},
    "pele": {"dilation_cap": 5, "feather_cap": 9, "diff_threshold": 6, "growth_padding": 12},
    "cabelo": {"dilation_cap": 8, "feather_cap": 15, "diff_threshold": 7, "growth_padding": 16},
}
REGION_KEYWORDS = {
    "sobrancelhas": ("eyebrow", "eyebrows", "sobrancelha", "sobrancelhas"),
    "olhos": ("eye", "eyes", "olho", "olhos", "eyelash", "eyelashes", "pestana", "pestanas"),
    "boca": ("mouth", "lips", "lip", "boca", "labio", "labios", "lipstick", "batom"),
    "nariz": ("nose", "nariz"),
    "orelhas": ("ear", "ears", "orelha", "orelhas"),
    "pescoco": ("neck", "pescoco", "pescoço"),
    "pele": ("skin", "pele", "face", "rosto", "cheek", "cheeks", "bochecha", "bochechas"),
    "cabelo": ("hair", "cabelo", "cabelos", "bangs", "franja", "franjas"),
}
COLOR_KEYWORDS = (
    "green",
    "blue",
    "brown",
    "hazel",
    "gray",
    "grey",
    "black",
    "white",
    "blond",
    "blonde",
    "red",
    "pink",
    "purple",
    "orange",
    "yellow",
    "gold",
    "silver",
    "verde",
    "azul",
    "castanho",
    "cinzento",
    "cinza",
    "preto",
    "branco",
    "loiro",
    "loira",
    "ruivo",
    "ruiva",
    "vermelho",
    "vermelha",
    "rosa",
    "roxo",
    "roxa",
    "dourado",
    "dourada",
)
COLOR_NAME_MAP = {
    "green": "green",
    "verde": "green",
    "verdes": "green",
    "blue": "blue",
    "azul": "blue",
    "azuis": "blue",
    "brown": "brown",
    "castanho": "brown",
    "castanhos": "brown",
    "castanhas": "brown",
    "hazel": "hazel",
    "gray": "gray",
    "grey": "gray",
    "cinzento": "gray",
    "cinza": "gray",
    "cinzentos": "gray",
    "cinzentas": "gray",
    "black": "black",
    "preto": "black",
    "pretos": "black",
    "pretas": "black",
    "white": "white",
    "branco": "white",
    "brancos": "white",
    "brancas": "white",
    "red": "red",
    "vermelho": "red",
    "vermelha": "red",
    "vermelhos": "red",
    "vermelhas": "red",
    "pink": "pink",
    "rosa": "pink",
    "rosas": "pink",
    "purple": "purple",
    "roxo": "purple",
    "roxa": "purple",
    "roxos": "purple",
    "roxas": "purple",
    "blond": "blond",
    "blonde": "blond",
    "loiro": "blond",
    "loira": "blond",
    "loiros": "blond",
    "loiras": "blond",
    "gold": "gold",
    "golden": "gold",
    "dourado": "gold",
    "dourada": "gold",
    "dourados": "gold",
    "douradas": "gold",
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
DECREASE_KEYWORDS = (
    "smaller",
    "smaller",
    "small",
    "thinner",
    "thin",
    "narrower",
    "less",
    "shorter",
    "decrease",
    "reduce",
    "reduced",
    "shrink",
    "minor",
    "pequeno",
    "pequena",
    "pequenos",
    "pequenas",
    "mais pequeno",
    "mais pequena",
    "mais pequenos",
    "mais pequenas",
    "fino",
    "fina",
    "finos",
    "finas",
    "mais fino",
    "mais fina",
    "menos",
    "menor",
    "menores",
    "afinar",
    "diminuir",
    "diminuido",
    "diminuida",
    "reduzir",
    "reduzido",
    "reduzida",
    "encolher",
    "encolhido",
    "encolhida",
    "curto",
    "curta",
    "curtos",
    "curtas",
)
INCREASE_KEYWORDS = (
    "larger",
    "large",
    "bigger",
    "broad",
    "fuller",
    "full",
    "thicker",
    "thick",
    "wider",
    "more",
    "longer",
    "grande",
    "grandes",
    "maior",
    "maiores",
    "mais grande",
    "mais grandes",
    "cheio",
    "cheia",
    "cheios",
    "cheias",
    "mais cheio",
    "mais cheia",
    "grosso",
    "grossa",
    "grossos",
    "grossas",
    "mais grosso",
    "mais grossa",
    "largo",
    "larga",
    "largos",
    "largas",
    "comprido",
    "comprida",
    "compridos",
    "compridas",
    "longo",
    "longa",
    "longos",
    "longas",
)


def save_rgb_image(path: Path, image_rgb) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def save_bgr_image(path: Path, image_bgr, cv2) -> None:
    save_rgb_image(path, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def build_soft_mask(mask_uint8, feather: int, cv2, np):
    if feather > 0:
        blur_size = feather * 2 + 1
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 0)
    return (mask_uint8.astype(np.float32) / 255.0)[..., None]


def extract_mask_outline_contours(mask_uint8, cv2, np, simplify_epsilon_ratio: float = 0.004):
    hard = (mask_uint8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 4.0:
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


def build_difference_image(before_bgr, after_bgr, mask_uint8, cv2, np, outside: bool):
    diff = cv2.absdiff(after_bgr, before_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if outside:
        keep = (mask_uint8 <= 0).astype(np.uint8)
    else:
        keep = (mask_uint8 > 0).astype(np.uint8)
    diff_gray = (diff_gray * keep).astype(np.uint8)
    return cv2.applyColorMap(diff_gray, cv2.COLORMAP_INFERNO)


def compute_diff_stats(before_bgr, after_bgr, mask_uint8, np) -> dict[str, object]:
    diff = np.abs(after_bgr.astype(np.int16) - before_bgr.astype(np.int16))
    max_per_pixel = diff.max(axis=2)
    inside = mask_uint8 > 0
    outside = ~inside

    def _stats(region):
        if not np.any(region):
            return {"pixels": 0, "changed_pixels": 0, "max_diff": 0, "mean_diff_changed": 0.0}
        changed = max_per_pixel[region] > 0
        changed_values = diff[region][changed]
        return {
            "pixels": int(region.sum()),
            "changed_pixels": int(changed.sum()),
            "max_diff": int(max_per_pixel[region].max()) if np.any(region) else 0,
            "mean_diff_changed": float(changed_values.mean()) if changed_values.size else 0.0,
        }

    return {
        "inside_mask": _stats(inside),
        "outside_mask": _stats(outside),
    }


def build_edge_falloff_mask(height: int, width: int, feather: int, np):
    if feather <= 0:
        return np.ones((height, width, 1), dtype=np.float32)

    y_indices, x_indices = np.indices((height, width))
    distance_to_border = np.minimum.reduce(
        [
            x_indices,
            y_indices,
            width - 1 - x_indices,
            height - 1 - y_indices,
        ]
    ).astype(np.float32)
    return np.clip(distance_to_border / float(feather), 0.0, 1.0)[..., None]


def warp_image_with_inverse_matrix(image_bgr, inverse_matrix, output_size, cv2):
    return cv2.warpAffine(
        image_bgr,
        inverse_matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def warp_mask_with_inverse_matrix(mask_uint8, inverse_matrix, output_size, cv2):
    return cv2.warpAffine(
        mask_uint8,
        inverse_matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().replace(",", " , ").split())


def split_prompt_segments(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    segments = [
        segment.strip()
        for segment in re.split(r"\s*(?:,|\band\b|\be\b)\s*", normalized)
        if segment.strip()
    ]
    return segments


def contains_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None


def detect_regions_in_segment(segment: str, target_regions: list[str]) -> list[str]:
    matched = []
    for region_name in target_regions:
        for keyword in REGION_KEYWORDS.get(region_name, ()):
            if contains_keyword(segment, keyword):
                matched.append(region_name)
                break
    return matched


def infer_region_intents(description: str, target_regions: list[str]) -> dict[str, str]:
    intents = {region_name: "generic" for region_name in target_regions}
    segments = split_prompt_segments(description)
    if not segments:
        return intents

    for segment in segments:
        matched_regions = detect_regions_in_segment(segment, target_regions)
        if not matched_regions:
            continue

        segment_intent = "generic"
        if any(keyword in segment for keyword in COLOR_KEYWORDS):
            segment_intent = "color"
        elif any(keyword in segment for keyword in DECREASE_KEYWORDS):
            segment_intent = "decrease"
        elif any(keyword in segment for keyword in INCREASE_KEYWORDS):
            segment_intent = "increase"

        for region_name in matched_regions:
            intents[region_name] = segment_intent

    return intents


def infer_region_attributes(description: str, target_regions: list[str]) -> dict[str, dict[str, object]]:
    attributes = {
        region_name: {"intent": "generic", "color": None, "focus": None}
        for region_name in target_regions
    }
    segments = split_prompt_segments(description)
    if not segments:
        return attributes

    for segment in segments:
        matched_regions = detect_regions_in_segment(segment, target_regions)
        if not matched_regions:
            continue

        segment_color = None
        for color_word, canonical_color in COLOR_NAME_MAP.items():
            if contains_keyword(segment, color_word):
                segment_color = canonical_color
                break

        segment_intent = "generic"
        if segment_color is not None:
            segment_intent = "color"
        elif any(contains_keyword(segment, keyword) for keyword in DECREASE_KEYWORDS):
            segment_intent = "decrease"
        elif any(contains_keyword(segment, keyword) for keyword in INCREASE_KEYWORDS):
            segment_intent = "increase"

        for region_name in matched_regions:
            attributes[region_name]["intent"] = segment_intent
            if segment_color is not None:
                attributes[region_name]["color"] = segment_color
            if region_name == "boca":
                if any(contains_keyword(segment, word) for word in ("lip", "lips", "labio", "labios", "lipstick", "batom")):
                    attributes[region_name]["focus"] = "lips"
                elif any(contains_keyword(segment, word) for word in ("mouth", "boca")):
                    attributes[region_name]["focus"] = "mouth"
            elif region_name == "nariz":
                attributes[region_name]["focus"] = "nose"

    return attributes


def get_region_mask_config(region_name: str, requested_dilation: int, requested_feather: int) -> dict[str, int]:
    tuning = REGION_MASK_TUNING.get(region_name, {})
    dilation = min(requested_dilation, tuning.get("dilation_cap", requested_dilation))
    feather = min(requested_feather, tuning.get("feather_cap", requested_feather))
    return {
        "dilation": dilation,
        "feather": feather,
        "diff_threshold": tuning.get("diff_threshold", 6),
        "growth_padding": tuning.get("growth_padding", 12),
    }


def build_difference_mask(original_bgr, edited_bgr, threshold: int, cv2, np):
    diff = cv2.absdiff(edited_bgr, original_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_mask = (diff_gray >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff_mask = cv2.dilate(diff_mask, kernel, iterations=1)
    return diff_mask


def apply_color_tint(base_bgr, mask_uint8, color_name: str, strength: float, cv2, np, feather: int = 3):
    target_bgr = COLOR_BGR_MAP.get(color_name)
    if target_bgr is None or not np.any(mask_uint8):
        return base_bgr

    alpha = build_soft_mask(mask_uint8, feather, cv2, np) * float(max(0.0, min(strength, 1.0)))
    light_hair_colors = {"blond", "gold", "white", "gray", "silver"}
    if color_name in light_hair_colors:
        hair_alpha = alpha[..., 0]
        base_float = base_bgr.astype(np.float32)
        gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.1)
        tone = np.clip(gray * 1.35 + 0.18, 0.0, 1.0)[..., None]

        if color_name in {"gray", "silver", "white"}:
            shadow = np.array([92, 92, 88], dtype=np.float32).reshape(1, 1, 3)
            highlight = np.array([224, 224, 216], dtype=np.float32).reshape(1, 1, 3)
        else:
            shadow = np.array([62, 96, 135], dtype=np.float32).reshape(1, 1, 3)
            highlight = np.array([120, 205, 246], dtype=np.float32).reshape(1, 1, 3)

        target_ramp = shadow * (1.0 - tone) + highlight * tone
        target_ramp = np.clip(target_ramp + detail[..., None] * 42.0, 0, 255)
        lightened = target_ramp * 0.82 + base_float * 0.18
        blended = (
            lightened * hair_alpha[..., None]
            + base_float * (1.0 - hair_alpha[..., None])
        ).clip(0, 255).astype(np.uint8)
        return blended

    hsv_image = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(np.uint8([[list(target_bgr)]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)

    recolored_hsv = hsv_image.copy()
    hue_alpha = alpha[..., 0]
    target_saturation = float(target_hsv[1])
    target_value = float(target_hsv[2])

    recolored_hsv[..., 0] = hsv_image[..., 0] * (1.0 - hue_alpha) + target_hsv[0] * hue_alpha

    recolored_hsv[..., 1] = np.clip(
        hsv_image[..., 1] * (1.0 - alpha[..., 0]) + target_saturation * alpha[..., 0],
        0,
        255,
    )
    recolored_hsv[..., 2] = np.clip(
        hsv_image[..., 2] * (1.0 - alpha[..., 0]) + target_value * alpha[..., 0],
        0,
        255,
    )

    recolored_bgr = cv2.cvtColor(recolored_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blended = (
        recolored_bgr.astype(np.float32) * alpha
        + base_bgr.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)
    return blended


def apply_eye_color_replacement(base_bgr, iris_mask_uint8, color_name: str, strength: float, cv2, np, feather: int = 1):
    target_bgr = COLOR_BGR_MAP.get(color_name)
    if target_bgr is None or not np.any(iris_mask_uint8):
        return base_bgr

    alpha = build_soft_mask(iris_mask_uint8, feather, cv2, np)
    alpha *= float(max(0.0, min(strength, 1.0)))
    alpha_2d = alpha[..., 0]

    hsv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(np.uint8([[list(target_bgr)]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)

    desired = hsv.copy()
    desired[..., 0] = target_hsv[0]
    desired[..., 1] = np.maximum(hsv[..., 1] * 0.55 + target_hsv[1] * 0.95, 185.0)
    desired[..., 2] = np.clip(np.maximum(hsv[..., 2] * 1.22 + 28.0, 95.0), 0, 255)

    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pupil_or_shadow = ((gray < 42) & (iris_mask_uint8 > 0)).astype(np.float32)
    highlight = ((gray > 205) & (iris_mask_uint8 > 0)).astype(np.float32)
    preserve = np.clip(pupil_or_shadow + highlight, 0.0, 1.0)
    alpha_2d = alpha_2d * (1.0 - preserve * 0.88)

    desired_bgr = cv2.cvtColor(desired.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    base_float = base_bgr.astype(np.float32)
    replaced = (
        desired_bgr * alpha_2d[..., None]
        + base_float * (1.0 - alpha_2d[..., None])
    ).clip(0, 255).astype(np.uint8)
    return replaced


def build_eye_recolor_mask(original_crop_bgr, base_eye_mask, cv2, np):
    if not np.any(base_eye_mask):
        return base_eye_mask
    height, width = base_eye_mask.shape[:2]
    eye_core = np.zeros_like(base_eye_mask)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((base_eye_mask > 0).astype(np.uint8), 8)
    for component_idx in range(1, component_count):
        x, y, w, h, area = stats[component_idx]
        if area < 20:
            continue
        center = (int(x + w / 2.0), int(y + h / 2.0))
        axes = (
            max(3, int(round(w * 0.32))),
            max(3, int(round(h * 0.42))),
        )
        component_mask = np.zeros_like(base_eye_mask)
        cv2.ellipse(component_mask, center, axes, 0, 0, 360, 255, -1)
        eye_core = cv2.max(eye_core, cv2.bitwise_and(component_mask, base_eye_mask))

    if not np.any(eye_core):
        eye_core = cv2.erode(
            base_eye_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

    gray = cv2.cvtColor(original_crop_bgr, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(original_crop_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    iris_candidates = (
        (gray > 25)
        & (gray < 170)
        & (saturation > 15)
    ).astype(np.uint8) * 255
    filtered = cv2.bitwise_and(eye_core, iris_candidates)
    if np.any(filtered):
        filtered = cv2.dilate(
            filtered,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        filtered = cv2.bitwise_and(filtered, eye_core)
        return filtered
    return eye_core


def build_iris_mask(original_crop_bgr, base_eye_mask, cv2, np):
    return build_eye_recolor_mask(original_crop_bgr, base_eye_mask, cv2, np)


def shrink_eyebrows_with_inpaint(base_bgr, eyebrow_mask, cv2, np):
    if not np.any(eyebrow_mask):
        return base_bgr, eyebrow_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(eyebrow_mask, kernel, iterations=1)
    edge_ring = cv2.subtract(eyebrow_mask, inner_mask)
    if not np.any(edge_ring):
        return base_bgr, eyebrow_mask

    inpainted = cv2.inpaint(base_bgr, edge_ring, 3, cv2.INPAINT_TELEA)
    refined = base_bgr.copy()
    refined[edge_ring > 0] = inpainted[edge_ring > 0]
    return refined, inner_mask


def compute_mask_bbox(mask_uint8, np):
    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(box, pad_x, pad_y, width, height):
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def scale_masked_region(base_bgr, target_mask, scale_x, scale_y, feather, cv2, np):
    bbox = compute_mask_bbox(target_mask, np)
    if bbox is None:
        return base_bgr, target_mask

    image_h, image_w = target_mask.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = max(6, int((x2 - x1) * 0.35))
    pad_y = max(6, int((y2 - y1) * 0.35))
    rx1, ry1, rx2, ry2 = expand_box((x1, y1, x2, y2), pad_x, pad_y, image_w, image_h)

    roi = base_bgr[ry1:ry2, rx1:rx2].copy()
    roi_mask = target_mask[ry1:ry2, rx1:rx2].copy()
    object_bbox = compute_mask_bbox(roi_mask, np)
    if object_bbox is None:
        return base_bgr, target_mask

    ox1, oy1, ox2, oy2 = object_bbox
    obj_patch = roi[oy1:oy2, ox1:ox2].copy()
    obj_mask = roi_mask[oy1:oy2, ox1:ox2].copy()

    new_w = max(2, int(round((ox2 - ox1) * scale_x)))
    new_h = max(2, int(round((oy2 - oy1) * scale_y)))
    scaled_patch = cv2.resize(obj_patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    cx = (ox1 + ox2) / 2.0
    cy = (oy1 + oy2) / 2.0
    nx1 = int(round(cx - new_w / 2.0))
    ny1 = int(round(cy - new_h / 2.0))
    nx2 = nx1 + new_w
    ny2 = ny1 + new_h

    canvas_patch = np.zeros_like(roi)
    canvas_mask = np.zeros_like(roi_mask)

    dest_x1 = max(0, nx1)
    dest_y1 = max(0, ny1)
    dest_x2 = min(roi.shape[1], nx2)
    dest_y2 = min(roi.shape[0], ny2)
    if dest_x2 <= dest_x1 or dest_y2 <= dest_y1:
        return base_bgr, target_mask

    src_x1 = max(0, -nx1)
    src_y1 = max(0, -ny1)
    src_x2 = src_x1 + (dest_x2 - dest_x1)
    src_y2 = src_y1 + (dest_y2 - dest_y1)

    canvas_patch[dest_y1:dest_y2, dest_x1:dest_x2] = scaled_patch[src_y1:src_y2, src_x1:src_x2]
    canvas_mask[dest_y1:dest_y2, dest_x1:dest_x2] = scaled_mask[src_y1:src_y2, src_x1:src_x2]

    alpha = build_soft_mask(canvas_mask, feather, cv2, np)
    blended_roi = (
        canvas_patch.astype(np.float32) * alpha
        + roi.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)

    result = base_bgr.copy()
    result[ry1:ry2, rx1:rx2] = blended_roi

    full_mask = np.zeros_like(target_mask)
    full_mask[ry1:ry2, rx1:rx2] = canvas_mask
    return result, full_mask


def get_region_target_labels(region_name: str, attributes: dict[str, object] | None):
    if region_name == "boca" and (attributes or {}).get("focus") == "lips":
        return [12, 13]
    return REGION_LABELS[region_name]


def uses_direct_attribute_replacement(region_name: str, attributes: dict[str, object]) -> bool:
    intent = attributes.get("intent")
    color_name = attributes.get("color")
    focus = attributes.get("focus")
    return (
        (region_name == "olhos" and color_name is not None)
        or (region_name == "boca" and color_name is not None)
        or (region_name == "sobrancelhas" and intent == "decrease")
        or (region_name == "boca" and intent == "increase" and focus == "lips")
        or (region_name == "nariz" and intent == "decrease")
    )


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "sim", "s", "on"}:
        return True
    if normalized in {"0", "false", "no", "nao", "não", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano invalido: {value}")


EDIT_REGION_ALIASES = {
    "hair": "hair",
    "cabelo": "hair",
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
    "beard": "beard",
    "barba": "beard",
    "mustache": "beard",
    "bigode": "beard",
    "goatee": "beard",
    "cavanhaque": "beard",
    "lower_face": "beard",
    "eyes": "eyes",
    "eye": "eyes",
    "olhos": "eyes",
    "olho": "eyes",
    "iris": "iris",
    "irises": "iris",
    "íris": "iris",
    "glasses": "eyes",
    "oculos": "eyes",
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
    "pescoço": "neck",
}


def resolve_edit_attribute(attribute: str, description: str, target_regions: list[str]) -> str:
    requested = (attribute or "auto").strip().lower()
    if requested != "auto":
        return EDIT_REGION_ALIASES.get(requested, requested)

    normalized = normalize_text(description)
    prompt_rules = (
        (("hair", "bangs", "cabelo", "franja"), "hair"),
        (("smile", "smiling", "mouth", "lips", "lipstick", "sorriso", "boca", "labios", "batom"), "mouth"),
        (("beard", "mustache", "goatee", "facial hair", "barba", "bigode", "cavanhaque"), "beard"),
        (("glasses", "eyes", "eye", "eyelashes", "olhos", "oculos", "pestanas"), "eyes"),
        (("iris", "irises", "íris"), "iris"),
        (("eyebrow", "eyebrows", "brows", "sobrancelha", "sobrancelhas"), "eyebrows"),
        (("nose", "nariz"), "nose"),
        (("ear", "ears", "orelha", "orelhas"), "ears"),
        (("neck", "pescoco", "pescoço"), "neck"),
        (("age", "older", "younger", "skin", "wrinkles", "freckles", "acne", "face", "idade", "pele", "rugas", "sardas"), "face"),
    )
    for keywords, region in prompt_rules:
        if any(contains_keyword(normalized, keyword) for keyword in keywords):
            return region

    region_priority = (
        ("cabelo", "hair"),
        ("boca", "mouth"),
        ("olhos", "eyes"),
        ("sobrancelhas", "eyebrows"),
        ("nariz", "nose"),
        ("orelhas", "ears"),
        ("pescoco", "neck"),
        ("pele", "face"),
    )
    for region_name, edit_attribute in region_priority:
        if region_name in target_regions:
            return edit_attribute

    return "face"


def create_edit_mask(attribute: str, parsing_map, cv2, np, dilation: int = 0):
    attribute = EDIT_REGION_ALIASES.get((attribute or "auto").lower(), (attribute or "face").lower())

    if attribute == "hair":
        labels = REGION_LABELS["cabelo"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "mouth":
        labels = REGION_LABELS["boca"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "face":
        labels = REGION_LABELS["pele"] + REGION_LABELS["nariz"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "beard":
        labels = REGION_LABELS["pele"]
        skin_mask = build_region_mask(parsing_map, labels, dilation, cv2, np)
        height, width = skin_mask.shape[:2]
        lower_face_mask = np.zeros_like(skin_mask)
        lower_face_mask[int(height * 0.52):height, :] = 255
        mouth_keepout = build_region_mask(parsing_map, REGION_LABELS["boca"], max(0, dilation // 2), cv2, np)
        beard_mask = cv2.bitwise_and(skin_mask, lower_face_mask)
        beard_mask = cv2.bitwise_and(beard_mask, cv2.bitwise_not(mouth_keepout))
        return beard_mask, labels

    if attribute == "eyes":
        labels = REGION_LABELS["olhos"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "iris":
        labels = REGION_LABELS["olhos"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "eyebrows":
        labels = REGION_LABELS["sobrancelhas"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "nose":
        labels = REGION_LABELS["nariz"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "ears":
        labels = REGION_LABELS["orelhas"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    if attribute == "neck":
        labels = REGION_LABELS["pescoco"]
        return build_region_mask(parsing_map, labels, dilation, cv2, np), labels

    labels = REGION_LABELS["pele"]
    return build_region_mask(parsing_map, labels, dilation, cv2, np), labels


EDIT_REGION_MASK_DEFAULTS = {
    "hair": {"dilation": 8, "blur": 15},
    "mouth": {"dilation": 1, "blur": 4},
    "face": {"dilation": 4, "blur": 9},
    "beard": {"dilation": 3, "blur": 7},
    "eyes": {"dilation": 0, "blur": 2},
    "iris": {"dilation": 0, "blur": 1},
    "eyebrows": {"dilation": 1, "blur": 3},
    "nose": {"dilation": 0, "blur": 0},
    "ears": {"dilation": 1, "blur": 4},
    "neck": {"dilation": 2, "blur": 6},
}


def get_edit_mask_defaults(attribute: str) -> dict[str, int]:
    attribute = EDIT_REGION_ALIASES.get((attribute or "face").lower(), (attribute or "face").lower())
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


def blend_with_original(original_bgr, edited_bgr, mask_uint8, np):
    alpha = (mask_uint8.astype(np.float32) / 255.0)[..., None]
    return (
        edited_bgr.astype(np.float32) * alpha
        + original_bgr.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)


def apply_inverse_warp(final_aligned_bgr, original_image_bgr, mask_uint8, bbox, alignment, cv2, np):
    image_h, image_w = original_image_bgr.shape[:2]
    if alignment.get("enabled") and alignment.get("inverse_matrix"):
        inverse_matrix = np.array(alignment["inverse_matrix"], dtype=np.float32)
        output_size = (image_w, image_h)
        warped_final = warp_image_with_inverse_matrix(final_aligned_bgr, inverse_matrix, output_size, cv2)
        warped_mask = warp_mask_with_inverse_matrix(mask_uint8, inverse_matrix, output_size, cv2)
        return blend_with_original(original_image_bgr, warped_final, warped_mask, np), warped_mask

    x1, y1, x2, y2 = [int(v) for v in bbox]
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    resized_final = cv2.resize(final_aligned_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    output = original_image_bgr.copy()
    roi = output[y1:y2, x1:x2]
    output[y1:y2, x1:x2] = blend_with_original(roi, resized_final, resized_mask, np)
    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = resized_mask
    return output, full_mask


def apply_repaint_inpainting(final_aligned_bgr, mask_uint8, cv2, np, steps: int, strength: float):
    if steps <= 0 or strength <= 0 or not np.any(mask_uint8):
        return final_aligned_bgr, np.zeros_like(mask_uint8)

    radius = max(2, min(9, int(round(steps / 5))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    hard_mask = (mask_uint8 > 16).astype(np.uint8) * 255
    inner = cv2.erode(hard_mask, kernel, iterations=1)
    outer = cv2.dilate(hard_mask, kernel, iterations=1)
    edge_ring = cv2.subtract(outer, inner)
    if not np.any(edge_ring):
        return final_aligned_bgr, edge_ring

    inpainted = cv2.inpaint(final_aligned_bgr, edge_ring, radius, cv2.INPAINT_TELEA)
    ring_alpha = build_soft_mask(edge_ring, radius, cv2, np) * float(max(0.0, min(strength, 1.0)))
    refined = (
        inpainted.astype(np.float32) * ring_alpha
        + final_aligned_bgr.astype(np.float32) * (1.0 - ring_alpha)
    ).clip(0, 255).astype(np.uint8)
    return refined, edge_ring


def save_parsing_debug(output_dir: Path, parsing_map, masks: dict[str, object], cv2, np) -> dict[str, str]:
    paths: dict[str, str] = {}
    parsing_path = output_dir / "parsing_map.png"
    save_rgb_image(parsing_path, parsing_map.astype(np.uint8))
    paths["parsing_map"] = str(parsing_path)

    color_map = np.zeros((*parsing_map.shape, 3), dtype=np.uint8)
    color_map[..., 0] = ((parsing_map * 37) % 255).astype(np.uint8)
    color_map[..., 1] = ((parsing_map * 71) % 255).astype(np.uint8)
    color_map[..., 2] = ((parsing_map * 113) % 255).astype(np.uint8)
    color_path = output_dir / "parsing_color_map.png"
    save_rgb_image(color_path, color_map)
    paths["parsing_color_map"] = str(color_path)

    for name, mask in masks.items():
        path = output_dir / f"{name}.png"
        save_rgb_image(path, mask.astype(np.uint8))
        paths[name] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aplica uma edicao StyleCLIP apenas na regiao facial escolhida."
    )
    parser.add_argument("--crop-metadata", type=str, required=True, help="JSON do crop principal.")
    parser.add_argument("--edited-image", type=str, required=True, help="Imagem editada pelo StyleCLIP.")
    parser.add_argument("--output-dir", type=str, required=True, help="Diretorio dos resultados localizados.")
    parser.add_argument("--description", type=str, default="", help="Descricao textual ja resolvida para a edicao.")
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        nargs="+",
        choices=sorted(REGION_LABELS.keys()),
        help="Uma ou varias regioes faciais onde a alteracao deve ser aplicada.",
    )
    parser.add_argument("--dilation", type=int, default=8, help="Expansao extra da mascara.")
    parser.add_argument("--feather", type=int, default=15, help="Suavizacao da transicao da mascara.")
    parser.add_argument("--use-face-parsing", type=parse_bool, default=True, help="Ativa Face Parsing para criar mascaras semanticas.")
    parser.add_argument(
        "--edit-region",
        type=str,
        default="auto",
        choices=tuple(sorted({"auto", *EDIT_REGION_ALIASES.keys()})),
        help="Regiao semantica usada para a mascara final.",
    )
    parser.add_argument("--mask-dilation", type=int, default=-1, help="Dilation final da mascara selecionada.")
    parser.add_argument("--mask-erosion", type=int, default=0, help="Erosion final da mascara selecionada.")
    parser.add_argument("--mask-blur", type=int, default=-1, help="Blur/feather final da mascara selecionada.")
    parser.add_argument("--mask-threshold", type=int, default=1, help="Threshold aplicado a mascara selecionada.")
    parser.add_argument("--audit-debug", "--debug", dest="debug", type=parse_bool, default=False, help="Guarda imagens e validacoes de auditoria.")
    parser.add_argument("--use-repaint", type=parse_bool, default=False, help="Ativa inpainting/RePaint local nas bordas da mascara.")
    parser.add_argument("--repaint-steps", type=int, default=20, help="Controla largura/raio do refinamento de inpainting local.")
    parser.add_argument("--repaint-strength", type=float, default=0.35, help="Forca do refinamento de inpainting local.")
    parser.add_argument(
        "--edge-feather",
        type=int,
        default=32,
        help="Suavizacao adicional para evitar bordas visiveis no limite do crop.",
    )
    parser.add_argument(
        "--blend-strength",
        type=float,
        default=0.8,
        help="Forca maxima da edicao localizada. Valores mais baixos preservam mais a imagem original.",
    )
    parser.add_argument(
        "--direct-refinement-strength",
        type=float,
        default=-1.0,
        help="Forca dos refinamentos diretos por atributo. Se omitido, usa blend-strength.",
    )
    parser.add_argument(
        "--disable-direct-refinements",
        action="store_true",
        help="Usa apenas a imagem editada pelo StyleCLIP dentro da mascara, sem retoques diretos por cor/escala.",
    )
    parser.add_argument(
        "--save-extra-debug",
        action="store_true",
        help="Guarda overlays e aliases antigos alem dos outputs essenciais.",
    )
    args = parser.parse_args()

    crop_metadata_path = Path(args.crop_metadata).resolve()
    edited_image_path = Path(args.edited_image).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not crop_metadata_path.exists():
        raise FileNotFoundError(f"Metadados do crop nao encontrados: {crop_metadata_path}")
    if not edited_image_path.exists():
        raise FileNotFoundError(f"Imagem editada nao encontrada: {edited_image_path}")

    metadata = json.loads(crop_metadata_path.read_text(encoding="utf-8"))
    input_image_path = Path(metadata["input_image"]).resolve()
    crop_path = Path(metadata["crop_path"]).resolve()
    bbox = metadata["bbox"]
    alignment = metadata.get("alignment") or {"enabled": False}

    if not input_image_path.exists():
        raise FileNotFoundError(f"Imagem original nao encontrada: {input_image_path}")
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop original nao encontrado: {crop_path}")

    configure_runtime_warnings()
    ensure_directories(output_dir=output_dir)

    with suppress_known_stderr_noise():
        cv2, np, torch, _, init_parsing_model, img2tensor, normalize = load_required_modules()
        model = build_parsing_model(init_parsing_model)

        original_image_bgr = cv2.imread(str(input_image_path))
        original_crop_bgr = cv2.imread(str(crop_path))
        edited_crop_bgr = cv2.imread(str(edited_image_path))

        if original_image_bgr is None:
            raise FileNotFoundError(f"Nao foi possivel ler a imagem original: {input_image_path}")
        if original_crop_bgr is None:
            raise FileNotFoundError(f"Nao foi possivel ler o crop original: {crop_path}")
        if edited_crop_bgr is None:
            raise FileNotFoundError(f"Nao foi possivel ler a imagem editada: {edited_image_path}")

        crop_h, crop_w = original_crop_bgr.shape[:2]
        edited_crop_bgr = cv2.resize(edited_crop_bgr, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        original_parsing = parse_face_crop(original_crop_bgr, model, cv2, np, torch, img2tensor, normalize)
        edited_parsing = parse_face_crop(edited_crop_bgr, model, cv2, np, torch, img2tensor, normalize)

        target_regions = list(dict.fromkeys(args.region))
        selected_attribute = resolve_edit_attribute(args.edit_region, args.description, target_regions)
        if args.edit_region != "auto":
            region_override = {
                "hair": "cabelo",
                "cabelo": "cabelo",
                "mouth": "boca",
                "boca": "boca",
                "smile": "boca",
                "sorriso": "boca",
                "lips": "boca",
                "lip": "boca",
                "labios": "boca",
                "labio": "boca",
                "face": "pele",
                "pele": "pele",
                "skin": "pele",
                "age": "pele",
                "idade": "pele",
                "older": "pele",
                "younger": "pele",
                "beard": "pele",
                "barba": "pele",
                "mustache": "pele",
                "bigode": "pele",
                "goatee": "pele",
                "cavanhaque": "pele",
                "lower_face": "pele",
                "eyes": "olhos",
                "eye": "olhos",
                "olhos": "olhos",
                "olho": "olhos",
                "glasses": "olhos",
                "oculos": "olhos",
                "eyebrows": "sobrancelhas",
                "eyebrow": "sobrancelhas",
                "brows": "sobrancelhas",
                "sobrancelhas": "sobrancelhas",
                "sobrancelha": "sobrancelhas",
                "nose": "nariz",
                "nariz": "nariz",
                "ears": "orelhas",
                "ear": "orelhas",
                "orelhas": "orelhas",
                "orelha": "orelhas",
                "neck": "pescoco",
                "pescoco": "pescoco",
                "pescoço": "pescoco",
            }.get(args.edit_region, "pele")
            target_regions = [region_override]
        elif selected_attribute == "mouth" and "boca" not in target_regions:
            target_regions = ["boca"]
        elif selected_attribute == "beard" and "pele" not in target_regions:
            target_regions = ["pele"]
        elif selected_attribute == "eyes" and "olhos" not in target_regions:
            target_regions = ["olhos"]
        elif selected_attribute == "eyebrows" and "sobrancelhas" not in target_regions:
            target_regions = ["sobrancelhas"]
        elif selected_attribute == "nose" and "nariz" not in target_regions:
            target_regions = ["nariz"]
        elif selected_attribute == "ears" and "orelhas" not in target_regions:
            target_regions = ["orelhas"]
        elif selected_attribute == "neck" and "pescoco" not in target_regions:
            target_regions = ["pescoco"]
        elif selected_attribute == "face" and "pele" not in target_regions:
            target_regions = ["pele"]

        region_attributes = infer_region_attributes(args.description, target_regions)
        if (
            selected_attribute == "eyes"
            and any(
                region_name == "olhos" and region_attributes.get(region_name, {}).get("color") is not None
                for region_name in target_regions
            )
        ):
            selected_attribute = "iris"
        region_intents = {region_name: region_attributes[region_name]["intent"] for region_name in target_regions}
        region_mask_configs = {
            region_name: get_region_mask_config(region_name, args.dilation, args.feather)
            for region_name in target_regions
        }
        diff_masks = {
            region_name: build_difference_mask(
                original_crop_bgr,
                edited_crop_bgr,
                region_mask_configs[region_name]["diff_threshold"],
                cv2,
                np,
            )
            for region_name in target_regions
        }

        union_region = np.zeros((crop_h, crop_w), dtype=np.uint8)
        styleclip_region = np.zeros((crop_h, crop_w), dtype=np.uint8)
        effective_feather = 0
        region_debug = {}
        face_parsing_masks = {
            "hair_mask": build_region_mask(original_parsing, REGION_LABELS["cabelo"], 0, cv2, np),
            "face_mask": build_region_mask(original_parsing, REGION_LABELS["pele"], 0, cv2, np),
            "face_skin_mask": build_region_mask(original_parsing, REGION_LABELS["pele"], 0, cv2, np),
            "mouth_mask": build_region_mask(original_parsing, REGION_LABELS["boca"], 0, cv2, np),
            "eyes_mask": build_region_mask(original_parsing, REGION_LABELS["olhos"], 0, cv2, np),
            "eyebrows_mask": build_region_mask(original_parsing, REGION_LABELS["sobrancelhas"], 0, cv2, np),
            "nose_mask": build_region_mask(original_parsing, REGION_LABELS["nariz"], 0, cv2, np),
            "ears_mask": build_region_mask(original_parsing, REGION_LABELS["orelhas"], 0, cv2, np),
            "neck_mask": build_region_mask(original_parsing, REGION_LABELS["pescoco"], 0, cv2, np),
            "teeth_mask": build_region_mask(original_parsing, [11], 0, cv2, np),
            "clothes_mask": build_region_mask(original_parsing, [15, 16], 0, cv2, np),
            "iris_mask": np.zeros((crop_h, crop_w), dtype=np.uint8),
        }
        custom_nose_mask = build_custom_nose_mask(original_crop_bgr, cv2, np)
        if custom_nose_mask is not None:
            mouth_keepout = cv2.dilate(
                face_parsing_masks["mouth_mask"],
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            custom_nose_mask = cv2.bitwise_and(custom_nose_mask, cv2.bitwise_not(mouth_keepout))
            face_parsing_masks["nose_mask"] = custom_nose_mask
        lower_half = np.zeros((crop_h, crop_w), dtype=np.uint8)
        lower_half[int(crop_h * 0.52):crop_h, :] = 255
        face_parsing_masks["lower_face_mask"] = cv2.bitwise_and(face_parsing_masks["face_mask"], lower_half)
        face_parsing_masks["background_mask"] = cv2.bitwise_not(cv2.max(
            face_parsing_masks["face_mask"],
            cv2.max(face_parsing_masks["hair_mask"], face_parsing_masks["mouth_mask"]),
        ))

        for region_name in target_regions:
            attributes = region_attributes.get(region_name, {})
            target_labels = get_region_target_labels(region_name, attributes)
            mask_config = region_mask_configs[region_name]
            region_intent = region_intents.get(region_name, "generic")
            color_name = attributes.get("color")
            focus = attributes.get("focus")

            original_region = build_region_mask(original_parsing, target_labels, mask_config["dilation"], cv2, np)
            edited_region = build_region_mask(edited_parsing, target_labels, mask_config["dilation"], cv2, np)
            allowed_growth = build_region_mask(
                original_parsing,
                target_labels,
                mask_config["dilation"] + mask_config["growth_padding"],
                cv2,
                np,
            )
            edited_region_limited = cv2.bitwise_and(edited_region, allowed_growth)

            if region_intent == "color":
                candidate_region = original_region
            elif region_intent == "decrease":
                candidate_region = edited_region_limited
            elif region_intent == "increase":
                candidate_region = cv2.max(original_region, edited_region_limited)
            else:
                candidate_region = cv2.bitwise_and(cv2.max(original_region, edited_region_limited), allowed_growth)

            candidate_region = cv2.bitwise_and(candidate_region, diff_masks[region_name])
            if not np.any(candidate_region):
                candidate_region = original_region

            union_region = cv2.max(union_region, candidate_region)
            uses_direct_local_refinement = uses_direct_attribute_replacement(region_name, attributes)
            if not uses_direct_local_refinement:
                styleclip_region = cv2.max(styleclip_region, candidate_region)
                effective_feather = max(effective_feather, mask_config["feather"])
            region_debug[region_name] = {
                "intent": region_intent,
                "mask_config": mask_config,
                "uses_direct_local_refinement": uses_direct_local_refinement,
            }

        resolved_mask_dilation = args.dilation
        resolved_mask_blur = args.feather
        if args.use_face_parsing:
            selected_region_mask, selected_labels = create_edit_mask(
                selected_attribute,
                original_parsing,
                cv2,
                np,
                dilation=0,
            )
            if selected_attribute in {"eyes", "iris"}:
                selected_region_mask = build_iris_mask(original_crop_bgr, selected_region_mask, cv2, np)
                face_parsing_masks["iris_mask"] = selected_region_mask
            elif selected_attribute == "nose" and custom_nose_mask is not None:
                selected_region_mask = custom_nose_mask
                selected_labels = ["facemesh_custom_nose_region"]
            mask_defaults = get_edit_mask_defaults(selected_attribute)
            mask_dilation = args.mask_dilation if args.mask_dilation >= 0 else mask_defaults["dilation"]
            mask_blur = args.mask_blur if args.mask_blur >= 0 else mask_defaults["blur"]
            resolved_mask_dilation = mask_dilation
            resolved_mask_blur = mask_blur
            selected_region_mask = refine_mask(
                selected_region_mask,
                dilation=mask_dilation,
                erosion=args.mask_erosion,
                blur=0,
                threshold=args.mask_threshold,
                cv2=cv2,
                np=np,
            )
            union_region = selected_region_mask
            selected_direct_replacement = any(
                uses_direct_attribute_replacement(region_name, region_attributes.get(region_name, {}))
                for region_name in target_regions
            )
            styleclip_region = np.zeros_like(selected_region_mask) if selected_direct_replacement else selected_region_mask
            effective_feather = mask_blur
            target_labels = selected_labels

        target_labels = sorted(
            {
                label
                for region_name in target_regions
                for label in REGION_LABELS[region_name]
            }
        )
        if args.use_face_parsing:
            target_labels = selected_labels
        if np.any(styleclip_region):
            alpha = build_soft_mask(styleclip_region, effective_feather, cv2, np)
            alpha *= build_edge_falloff_mask(crop_h, crop_w, args.edge_feather, np)
            alpha *= float(max(0.0, min(args.blend_strength, 1.0)))
        else:
            alpha = np.zeros((crop_h, crop_w, 1), dtype=np.float32)

        placement_alpha = build_soft_mask(union_region, max(4, effective_feather), cv2, np)
        placement_alpha *= build_edge_falloff_mask(crop_h, crop_w, args.edge_feather, np)

        localized_crop_bgr = (
            edited_crop_bgr.astype(np.float32) * alpha
            + original_crop_bgr.astype(np.float32) * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)
        if selected_attribute == "nose":
            localized_crop_bgr = preserve_original_texture_in_region(
                localized_crop_bgr,
                original_crop_bgr,
                union_region,
                cv2,
                np,
            )

        direct_refinement_strength = (
            float(max(0.0, min(args.direct_refinement_strength, 1.0)))
            if args.direct_refinement_strength >= 0
            else float(max(0.0, min(args.blend_strength, 1.0)))
        )
        direct_refinements = {}
        nose_geometry_refinement = False
        nose_geometry_intent = resolve_nose_geometry_intent(args.description)
        if not args.disable_direct_refinements:
            for region_name in target_regions:
                attributes = region_attributes.get(region_name, {})
                base_labels = get_region_target_labels(region_name, attributes)
                if region_name == "nariz" and custom_nose_mask is not None:
                    precise_mask = custom_nose_mask
                else:
                    precise_mask = build_region_mask(original_parsing, base_labels, 0, cv2, np)
                intent = attributes.get("intent")
                color_name = attributes.get("color")
                focus = attributes.get("focus")

                if region_name == "olhos" and color_name:
                    eye_mask = build_iris_mask(original_crop_bgr, precise_mask, cv2, np)
                    face_parsing_masks["iris_mask"] = eye_mask
                    localized_crop_bgr = apply_eye_color_replacement(
                        localized_crop_bgr,
                        eye_mask,
                        color_name,
                        1.0,
                        cv2,
                        np,
                        feather=1,
                    )
                    direct_refinements[region_name] = {"mode": "direct_attribute_replacement", "color": color_name}
                    continue

                if region_name == "cabelo" and color_name:
                    refined_hair = apply_color_tint(
                        original_crop_bgr,
                        precise_mask,
                        color_name,
                        min(0.75, direct_refinement_strength),
                        cv2,
                        np,
                        feather=max(5, args.feather),
                    )
                    hair_alpha = build_soft_mask(precise_mask, max(5, args.feather), cv2, np)
                    localized_crop_bgr = (
                        refined_hair.astype(np.float32) * hair_alpha
                        + localized_crop_bgr.astype(np.float32) * (1.0 - hair_alpha)
                    ).clip(0, 255).astype(np.uint8)
                    direct_refinements[region_name] = {"mode": "direct_color", "color": color_name}
                    continue

                if region_name == "boca" and intent == "increase":
                    lip_mask = build_region_mask(original_parsing, [12, 13], 1, cv2, np) if focus == "lips" else precise_mask
                    refined_lips, scaled_mask = scale_masked_region(
                        localized_crop_bgr,
                        lip_mask,
                        scale_x=1.10,
                        scale_y=1.20,
                        feather=3,
                        cv2=cv2,
                        np=np,
                    )
                    lip_alpha = build_soft_mask(scaled_mask if np.any(scaled_mask) else lip_mask, 3, cv2, np)
                    localized_crop_bgr = (
                        refined_lips.astype(np.float32) * lip_alpha
                        + localized_crop_bgr.astype(np.float32) * (1.0 - lip_alpha)
                    ).clip(0, 255).astype(np.uint8)
                    direct_refinements[region_name] = {
                        "mode": "direct_scale",
                        "focus": focus or "mouth",
                        "scale_x": 1.10,
                        "scale_y": 1.20,
                    }
                    continue

                if region_name == "boca" and color_name:
                    lip_mask = build_region_mask(original_parsing, [12, 13], 1, cv2, np)
                    refined_lips = apply_color_tint(original_crop_bgr, lip_mask, color_name, 0.65, cv2, np, feather=3)
                    lip_alpha = build_soft_mask(lip_mask, 3, cv2, np)
                    localized_crop_bgr = (
                        refined_lips.astype(np.float32) * lip_alpha
                        + localized_crop_bgr.astype(np.float32) * (1.0 - lip_alpha)
                    ).clip(0, 255).astype(np.uint8)
                    direct_refinements[region_name] = {"mode": "direct_color", "color": color_name}
                    continue

                if region_name == "sobrancelhas" and intent == "decrease":
                    refined_brows, inner_brow_mask = shrink_eyebrows_with_inpaint(original_crop_bgr, precise_mask, cv2, np)
                    brow_alpha = build_soft_mask(precise_mask, 2, cv2, np)
                    localized_crop_bgr = (
                        refined_brows.astype(np.float32) * brow_alpha
                        + localized_crop_bgr.astype(np.float32) * (1.0 - brow_alpha)
                    ).clip(0, 255).astype(np.uint8)
                    direct_refinements[region_name] = {
                        "mode": "direct_shrink",
                        "remaining_mask_pixels": int((inner_brow_mask > 0).sum()),
                    }

                if region_name == "nariz" and nose_geometry_intent is not None:
                    nose_mask = precise_mask
                    refined_nose, scaled_mask = shrink_masked_region_with_inpaint(
                        original_crop_bgr,
                        nose_mask,
                        cv2=cv2,
                        np=np,
                        scale_x=nose_geometry_intent.scale_x,
                        scale_y=nose_geometry_intent.scale_y,
                        feather=5,
                        inpaint_radius=3,
                        center_y_ratio=nose_geometry_intent.center_y_ratio,
                        lower_extension_ratio=nose_geometry_intent.lower_extension_ratio,
                        keepout_mask=face_parsing_masks.get("mouth_mask"),
                    )
                    nose_alpha = build_soft_mask(scaled_mask if np.any(scaled_mask) else nose_mask, 1, cv2, np)
                    localized_crop_bgr = (
                        refined_nose.astype(np.float32) * nose_alpha
                        + original_crop_bgr.astype(np.float32) * (1.0 - nose_alpha)
                    ).clip(0, 255).astype(np.uint8)
                    nose_geometry_refinement = True
                    direct_refinements[region_name] = {
                        "mode": "direct_nose_smooth_warp",
                        "focus": "nose",
                        "scale_x": nose_geometry_intent.scale_x,
                        "scale_y": nose_geometry_intent.scale_y,
                        "center_y_ratio": nose_geometry_intent.center_y_ratio,
                        "lower_extension_ratio": nose_geometry_intent.lower_extension_ratio,
                        "intent": nose_geometry_intent.reason,
                        "uses_inpaint_background": True,
                    }

        if selected_attribute == "nose" and not nose_geometry_refinement:
            localized_crop_bgr = preserve_original_texture_in_region(
                localized_crop_bgr,
                original_crop_bgr,
                union_region,
                cv2,
                np,
            )

        mask_color = np.zeros_like(original_crop_bgr)
        mask_color[..., 1] = union_region
        overlay_bgr = cv2.addWeighted(localized_crop_bgr, 0.7, mask_color, 0.3, 0)

        final_repainted_aligned_bgr, repaint_edge_mask = apply_repaint_inpainting(
            localized_crop_bgr,
            union_region,
            cv2,
            np,
            steps=args.repaint_steps if args.use_repaint else 0,
            strength=args.repaint_strength if args.use_repaint else 0.0,
        )
        final_aligned_for_warp = final_repainted_aligned_bgr if args.use_repaint else localized_crop_bgr
        final_mask_for_warp = refine_mask(
            union_region,
            dilation=0,
            erosion=0,
            blur=max(0, effective_feather),
            threshold=args.mask_threshold,
            cv2=cv2,
            np=np,
        )

        image_h, image_w = original_image_bgr.shape[:2]
        localized_on_image_bgr = original_image_bgr.copy()
        region_map_on_image_bgr = original_image_bgr.copy()
        full_mask_on_image_uint8 = np.zeros((image_h, image_w), dtype=np.uint8)

        if alignment.get("enabled") and alignment.get("inverse_matrix"):
            localized_on_image_bgr, full_mask_on_image_uint8 = apply_inverse_warp(
                final_aligned_for_warp,
                original_image_bgr,
                final_mask_for_warp,
                bbox,
                alignment,
                cv2,
                np,
            )
            inverse_matrix = np.array(alignment["inverse_matrix"], dtype=np.float32)
            output_size = (image_w, image_h)
            warped_overlay = warp_image_with_inverse_matrix(overlay_bgr, inverse_matrix, output_size, cv2)
            full_alpha = (full_mask_on_image_uint8.astype(np.float32) / 255.0)[..., None]
            region_map_on_image_bgr = (
                warped_overlay.astype(np.float32) * full_alpha
                + original_image_bgr.astype(np.float32) * (1.0 - full_alpha)
            ).clip(0, 255).astype(np.uint8)
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            target_w = x2 - x1
            target_h = y2 - y1

            crop_resized_for_image = final_aligned_for_warp
            overlay_resized_for_image = overlay_bgr
            alpha_resized_for_image = (final_mask_for_warp.astype(np.float32) / 255.0)[..., None]
            if final_aligned_for_warp.shape[0] != target_h or final_aligned_for_warp.shape[1] != target_w:
                crop_resized_for_image = cv2.resize(final_aligned_for_warp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                overlay_resized_for_image = cv2.resize(overlay_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                alpha_resized_for_image = cv2.resize(
                    final_mask_for_warp,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                alpha_resized_for_image = (alpha_resized_for_image.astype(np.float32) / 255.0)[..., None]

            original_roi = original_image_bgr[y1:y2, x1:x2].astype(np.float32)
            full_mask_on_image_uint8[y1:y2, x1:x2] = (
                alpha_resized_for_image[..., 0] * 255.0
            ).clip(0, 255).astype(np.uint8)
            localized_on_image_bgr[y1:y2, x1:x2] = (
                crop_resized_for_image.astype(np.float32) * alpha_resized_for_image
                + original_roi * (1.0 - alpha_resized_for_image)
            ).clip(0, 255).astype(np.uint8)
            region_map_on_image_bgr[y1:y2, x1:x2] = (
                overlay_resized_for_image.astype(np.float32) * alpha_resized_for_image
                + original_roi * (1.0 - alpha_resized_for_image)
            ).clip(0, 255).astype(np.uint8)

    aligned_original_path = output_dir / "aligned_original.png"
    original_image_copy_path = output_dir / "original_image.png"
    localized_crop_path = output_dir / "localized_crop.png"
    localized_mask_path = output_dir / "localized_mask.png"
    localized_overlay_path = output_dir / "localized_overlay.png"
    localized_on_image_path = output_dir / "localized_on_image.png"
    styleclip_edit_path = output_dir / "styleclip_edit.png"
    edit_mask_path = output_dir / "edit_mask.png"
    selected_edit_mask_path = output_dir / "selected_edit_mask.png"
    selected_mask_path = output_dir / "selected_mask.png"
    selected_mask_overlay_path = output_dir / "selected_mask_overlay.png"
    iris_overlay_path = output_dir / "iris_overlay.png"
    final_edit_mask_path = output_dir / "final_edit_mask.png"
    edit_mask_on_original_path = output_dir / "edit_mask_on_original.png"
    final_blended_aligned_path = output_dir / "final_blended_aligned.png"
    final_blended_path = output_dir / "final_blended.png"
    final_aligned_path = output_dir / "final_aligned.png"
    final_repainted_aligned_path = output_dir / "final_repainted_aligned.png"
    final_on_original_path = output_dir / "final_on_original.png"
    resultado_final_path = output_dir / "resultado_final.png"
    landmarks_path = output_dir / "landmarks_478_enumerados.png"
    inverse_warp_mask_path = output_dir / "inverse_warp_mask.png"
    difference_outside_mask_path = output_dir / "difference_outside_mask.png"
    difference_inside_mask_path = output_dir / "difference_inside_mask.png"
    repaint_input_path = output_dir / "repaint_input.png"
    repaint_mask_path = output_dir / "repaint_mask.png"
    repaint_output_path = output_dir / "repaint_output.png"
    validation_report_path = output_dir / "validation_report.json"
    prompt_region_map_path = output_dir / "prompt_region_map.png"
    styleclip_resized_path = output_dir / "styleclip_edited_crop.png"
    metadata_path = output_dir / "localized_edit_metadata.json"

    save_rgb_image(edit_mask_path, union_region)
    save_rgb_image(original_image_copy_path, cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB))
    save_rgb_image(final_on_original_path, cv2.cvtColor(localized_on_image_bgr, cv2.COLOR_BGR2RGB))
    save_rgb_image(resultado_final_path, cv2.cvtColor(localized_on_image_bgr, cv2.COLOR_BGR2RGB))
    facemesh_overlay_bgr = draw_numbered_facemesh_overlay(original_image_bgr, cv2, scale=2.0)
    if facemesh_overlay_bgr is not None:
        save_bgr_image(landmarks_path, facemesh_overlay_bgr, cv2)
    selected_overlay_bgr = build_mask_overlay(original_crop_bgr, final_mask_for_warp, cv2, np)
    save_bgr_image(selected_mask_overlay_path, selected_overlay_bgr, cv2)

    if args.debug:
        save_bgr_image(output_dir / "original.png", original_image_bgr, cv2)
        save_rgb_image(aligned_original_path, cv2.cvtColor(original_crop_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(styleclip_edit_path, cv2.cvtColor(edited_crop_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(selected_edit_mask_path, union_region)
        save_rgb_image(selected_mask_path, union_region)
        save_rgb_image(final_edit_mask_path, final_mask_for_warp)
        save_rgb_image(final_blended_aligned_path, cv2.cvtColor(localized_crop_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(final_blended_path, cv2.cvtColor(localized_crop_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(final_aligned_path, cv2.cvtColor(final_aligned_for_warp, cv2.COLOR_BGR2RGB))
        save_rgb_image(final_repainted_aligned_path, cv2.cvtColor(final_aligned_for_warp, cv2.COLOR_BGR2RGB))
        save_rgb_image(inverse_warp_mask_path, full_mask_on_image_uint8)

    parsing_debug_paths = save_parsing_debug(
        output_dir,
        original_parsing,
        {
            **face_parsing_masks,
            "selected_edit_mask": union_region,
            "selected_mask": union_region,
            "final_edit_mask": final_mask_for_warp,
            "inverse_warp_mask": full_mask_on_image_uint8,
        },
        cv2,
        np,
    ) if args.use_face_parsing and args.debug else {}

    if args.debug:
        if np.any(face_parsing_masks.get("iris_mask", np.zeros_like(union_region))):
            iris_overlay_bgr = build_mask_overlay(original_crop_bgr, face_parsing_masks["iris_mask"], cv2, np, color_bgr=(255, 80, 0), alpha=0.55)
            save_bgr_image(iris_overlay_path, iris_overlay_bgr, cv2)
        save_bgr_image(difference_outside_mask_path, build_difference_image(original_image_bgr, localized_on_image_bgr, full_mask_on_image_uint8, cv2, np, outside=True), cv2)
        save_bgr_image(difference_inside_mask_path, build_difference_image(original_image_bgr, localized_on_image_bgr, full_mask_on_image_uint8, cv2, np, outside=False), cv2)
        save_bgr_image(repaint_input_path, localized_crop_bgr, cv2)
        save_rgb_image(repaint_mask_path, repaint_edge_mask if args.use_repaint else union_region)
        save_bgr_image(repaint_output_path, final_aligned_for_warp, cv2)
    editable_region_outline_aligned = extract_mask_outline_contours(final_mask_for_warp, cv2, np)
    editable_region_outline_full = extract_mask_outline_contours(full_mask_on_image_uint8, cv2, np)
    validation_report = {
        "aligned_diff": compute_diff_stats(original_crop_bgr, final_aligned_for_warp, final_mask_for_warp, np),
        "full_image_diff": compute_diff_stats(original_image_bgr, localized_on_image_bgr, full_mask_on_image_uint8, np),
        "outside_mask_should_be_unchanged": True,
        "mask_pixels_aligned": int((final_mask_for_warp > 0).sum()),
        "mask_pixels_full": int((full_mask_on_image_uint8 > 0).sum()),
        "editable_region_outline_contours": len(editable_region_outline_aligned),
        "selected_attribute": selected_attribute,
        "target_regions": target_regions,
    }
    validation_report_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    if args.save_extra_debug:
        save_rgb_image(localized_crop_path, cv2.cvtColor(localized_crop_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(localized_mask_path, union_region)
        save_rgb_image(localized_overlay_path, cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(localized_on_image_path, cv2.cvtColor(localized_on_image_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(edit_mask_on_original_path, full_mask_on_image_uint8)
        save_rgb_image(output_dir / "repaint_edge_mask.png", repaint_edge_mask)
        save_rgb_image(prompt_region_map_path, cv2.cvtColor(region_map_on_image_bgr, cv2.COLOR_BGR2RGB))
        save_rgb_image(styleclip_resized_path, cv2.cvtColor(edited_crop_bgr, cv2.COLOR_BGR2RGB))

    localized_metadata = {
        "input_image": str(input_image_path),
        "crop_path": str(crop_path),
        "edited_image": str(edited_image_path),
        "aligned_original_path": str(aligned_original_path) if args.debug else None,
        "original_image_copy_path": str(original_image_copy_path),
        "region": target_regions,
        "region_labels": target_labels,
        "edit_region": args.edit_region,
        "description": args.description,
        "region_intents": region_intents,
        "region_attributes": region_attributes,
        "region_debug": region_debug,
        "direct_refinements": direct_refinements,
        "dilation": args.dilation,
        "feather": args.feather,
        "edge_feather": args.edge_feather,
        "blend_strength": args.blend_strength,
        "use_face_parsing": args.use_face_parsing,
        "mask_dilation": resolved_mask_dilation,
        "mask_erosion": args.mask_erosion,
        "mask_blur": resolved_mask_blur,
        "mask_threshold": args.mask_threshold,
        "use_repaint": args.use_repaint,
        "repaint_mode": "opencv_edge_inpainting",
        "repaint_steps": args.repaint_steps,
        "repaint_strength": args.repaint_strength,
        "direct_refinement_strength": direct_refinement_strength,
        "direct_refinements_enabled": not args.disable_direct_refinements,
        "bbox": bbox,
        "crop_coords": bbox,
        "alignment": alignment,
        "transform_matrix": alignment.get("matrix") if isinstance(alignment, dict) else None,
        "inverse_transform_matrix": alignment.get("inverse_matrix") if isinstance(alignment, dict) else None,
        "localized_crop_path": str(localized_crop_path) if args.save_extra_debug else (str(final_blended_aligned_path) if args.debug else None),
        "localized_mask_path": str(localized_mask_path) if args.save_extra_debug else str(edit_mask_path),
        "localized_overlay_path": str(localized_overlay_path) if args.save_extra_debug else None,
        "localized_on_image_path": str(localized_on_image_path) if args.save_extra_debug else str(final_on_original_path),
        "styleclip_edit_path": str(styleclip_edit_path) if args.debug else None,
        "edit_mask_path": str(edit_mask_path),
        "selected_edit_mask_path": str(selected_edit_mask_path) if args.debug else None,
        "selected_mask_path": str(selected_mask_path) if args.debug else str(edit_mask_path),
        "selected_mask_overlay_path": str(selected_mask_overlay_path),
        "iris_mask_path": parsing_debug_paths.get("iris_mask") if parsing_debug_paths else None,
        "iris_overlay_path": str(iris_overlay_path) if args.debug and np.any(face_parsing_masks.get("iris_mask", np.zeros_like(union_region))) else None,
        "final_edit_mask_path": str(final_edit_mask_path) if args.debug else str(edit_mask_path),
        "edit_mask_on_original_path": str(edit_mask_on_original_path) if args.save_extra_debug else None,
        "final_blended_aligned_path": str(final_blended_aligned_path) if args.debug else None,
        "final_blended_path": str(final_blended_path) if args.debug else None,
        "final_aligned_path": str(final_aligned_path) if args.debug else None,
        "final_repainted_aligned_path": str(final_repainted_aligned_path) if args.debug else None,
        "final_on_original_path": str(final_on_original_path),
        "resultado_final_path": str(resultado_final_path),
        "landmarks_path": str(landmarks_path) if landmarks_path.exists() else None,
        "editable_region_outline": {
            "region": selected_attribute,
            "source": "automatic_edit_mask",
            "coordinate_space": "aligned_crop",
            "contours": editable_region_outline_aligned,
            "full_image_coordinate_space": "original_image",
            "full_image_contours": editable_region_outline_full,
            "mask_path": str(edit_mask_path),
            "overlay_path": str(selected_mask_overlay_path),
        },
        "inverse_warp_mask_path": str(inverse_warp_mask_path) if args.debug else None,
        "difference_outside_mask_path": str(difference_outside_mask_path) if args.debug else None,
        "difference_inside_mask_path": str(difference_inside_mask_path) if args.debug else None,
        "repaint_input_path": str(repaint_input_path) if args.debug else None,
        "repaint_mask_path": str(repaint_mask_path) if args.debug else None,
        "repaint_output_path": str(repaint_output_path) if args.debug else None,
        "validation_report_path": str(validation_report_path),
        "prompt_region_map_path": str(prompt_region_map_path) if args.save_extra_debug else None,
        "parsing_debug_paths": parsing_debug_paths,
    }
    metadata_path.write_text(json.dumps(localized_metadata, indent=2), encoding="utf-8")

    print(f"Mascara de edicao guardada em: {edit_mask_path}")
    if args.debug:
        print(f"Crop original alinhado guardado em: {aligned_original_path}")
        print(f"Final alinhado guardado em: {final_blended_aligned_path}")
    if args.use_repaint and args.debug:
        print(f"Final com inpainting/RePaint local guardado em: {final_repainted_aligned_path}")
    print(f"Final na imagem original guardado em: {final_on_original_path}")
    print(f"Resultado final guardado em: {resultado_final_path}")
    if args.save_extra_debug:
        print(f"Mapa das regioes do prompt guardado em: {prompt_region_map_path}")
    print(f"Metadados guardados em: {metadata_path}")


if __name__ == "__main__":
    main()
