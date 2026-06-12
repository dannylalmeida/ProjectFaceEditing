from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NoseGeometryIntent:
    scale_x: float
    scale_y: float
    feather: int
    center_y_ratio: float
    lower_extension_ratio: float
    reason: str
    width_decrease: bool
    length_increase: bool
    length_decrease: bool


WIDTH_DECREASE_TERMS = (
    "smaller",
    "small",
    "thin",
    "thinner",
    "narrow",
    "narrower",
    "less wide",
    "decrease",
    "reduce",
    "reduced",
    "shrink",
    "menor",
    "menores",
    "pequeno",
    "pequena",
    "pequenos",
    "pequenas",
    "fino",
    "fina",
    "finos",
    "finas",
    "estreito",
    "estreita",
    "estreitos",
    "estreitas",
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
)

WIDTH_INCREASE_TERMS = (
    "larger",
    "bigger",
    "wide",
    "wider",
    "broad",
    "broader",
    "maior",
    "maiores",
    "largo",
    "larga",
    "largos",
    "largas",
)

LENGTH_INCREASE_TERMS = (
    "long",
    "longer",
    "more long",
    "comprido",
    "comprida",
    "compridos",
    "compridas",
    "longo",
    "longa",
    "longos",
    "longas",
    "mais comprido",
    "mais comprida",
    "mais longo",
    "mais longa",
)

LENGTH_DECREASE_TERMS = (
    "short",
    "shorter",
    "less long",
    "less longer",
    "curto",
    "curta",
    "curtos",
    "curtas",
    "mais curto",
    "mais curta",
    "mais curtos",
    "mais curtas",
    "menos comprido",
    "menos comprida",
    "menos compridos",
    "menos compridas",
    "menos longo",
    "menos longa",
    "menos longos",
    "menos longas",
)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return f" {ascii_text.lower().replace('-', ' ')} "


def _has_any_term(normalized_text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized_text) for term in terms)


def resolve_nose_geometry_intent(description: str) -> NoseGeometryIntent | None:
    normalized = _normalize_text(description)
    mentions_nose = _has_any_term(normalized, ("nose", "nariz"))
    width_decrease = _has_any_term(normalized, WIDTH_DECREASE_TERMS)
    width_increase = _has_any_term(normalized, WIDTH_INCREASE_TERMS)
    length_decrease = _has_any_term(normalized, LENGTH_DECREASE_TERMS)
    length_increase = _has_any_term(normalized, LENGTH_INCREASE_TERMS)
    if length_decrease:
        length_increase = False

    if not mentions_nose and not (width_decrease or width_increase or length_increase or length_decrease):
        return None
    if not (width_decrease or width_increase or length_increase or length_decrease):
        return None

    scale_x = 1.0
    scale_y = 1.0
    reasons: list[str] = []

    if width_decrease:
        scale_x = 0.78 if length_increase else 0.76
        scale_y = 0.95
        reasons.append("width_decrease")
    elif width_increase:
        scale_x = 1.13
        reasons.append("width_increase")

    if length_increase:
        scale_y = 1.24 if (width_decrease or width_increase) else 1.30
        reasons.append("length_increase")
    elif length_decrease:
        scale_y = 0.74 if (width_decrease or width_increase) else 0.70
        reasons.append("length_decrease")

    length_only = (length_increase or length_decrease) and not (width_decrease or width_increase)
    center_y_ratio = 0.34 if length_decrease else (0.24 if length_increase else 0.58)
    lower_extension_ratio = 0.34 if length_only and length_increase else (0.24 if length_increase else (0.16 if length_decrease else 0.0))
    return NoseGeometryIntent(
        scale_x=scale_x,
        scale_y=scale_y,
        feather=18 if length_only else (20 if length_increase else 18),
        center_y_ratio=center_y_ratio,
        lower_extension_ratio=lower_extension_ratio,
        reason="+".join(reasons),
        width_decrease=width_decrease,
        length_increase=length_increase,
        length_decrease=length_decrease,
    )


def compute_mask_bbox(mask_uint8, np):
    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(box, pad_x: int, pad_y: int, width: int, height: int):
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _inner_feather_mask(mask_uint8, feather: int, cv2, np):
    hard = (mask_uint8 > 0).astype(np.uint8)
    if not np.any(hard):
        return hard.astype(np.float32)
    if feather <= 0:
        return hard.astype(np.float32)

    distance = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(max(1, feather)), 0.0, 1.0)
    alpha[hard == 0] = 0.0
    return alpha.astype(np.float32)


def _odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _build_working_mask(target_mask, cv2, np, lower_extension_ratio: float, keepout_mask=None):
    working_mask = (target_mask > 0).astype(np.uint8) * 255
    bbox = compute_mask_bbox(working_mask, np)
    if bbox is None:
        return working_mask

    if lower_extension_ratio > 0:
        height, width = working_mask.shape[:2]
        x1, y1, x2, y2 = bbox
        mask_w = max(1, x2 - x1)
        mask_h = max(1, y2 - y1)
        extension_px = max(2, int(round(mask_h * float(lower_extension_ratio))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_odd_at_least(mask_w * 0.28), _odd_at_least(extension_px * 2 + 1)),
        )
        expanded = cv2.dilate(working_mask, kernel, iterations=1)
        lower_gate = np.zeros_like(working_mask)
        gate_y1 = max(0, int(round(y1 + mask_h * 0.34)))
        gate_y2 = min(height, y2 + extension_px)
        gate_x1 = max(0, x1 - int(round(mask_w * 0.24)))
        gate_x2 = min(width, x2 + int(round(mask_w * 0.24)))
        lower_gate[gate_y1:gate_y2, gate_x1:gate_x2] = 255
        working_mask = cv2.max(working_mask, cv2.bitwise_and(expanded, lower_gate))

    if keepout_mask is not None and np.any(keepout_mask):
        keepout = cv2.dilate(
            (keepout_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        working_mask = cv2.bitwise_and(working_mask, cv2.bitwise_not(keepout))

    return working_mask


def _expand_working_mask_for_geometry(target_mask, working_mask, cv2, np, scale_x: float, scale_y: float, feather: int, keepout_mask=None):
    bbox = compute_mask_bbox(target_mask, np)
    if bbox is None:
        return working_mask

    height, width = target_mask.shape[:2]
    x1, y1, x2, y2 = bbox
    mask_w = max(1, x2 - x1)
    mask_h = max(1, y2 - y1)

    pad_x = int(round(mask_w * (0.05 + abs(float(scale_x) - 1.0) * 0.75)))
    pad_top = int(round(mask_h * (0.02 + max(0.0, float(scale_y) - 1.0) * 0.35)))
    pad_bottom = int(round(mask_h * (0.05 + abs(float(scale_y) - 1.0) * 0.85)))
    pad_x = max(2, min(pad_x, int(round(mask_w * 0.20))))
    pad_top = max(1, min(pad_top, int(round(mask_h * 0.16))))
    pad_bottom = max(2, min(pad_bottom, int(round(mask_h * 0.22))))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_at_least(pad_x * 2 + 1), _odd_at_least(max(pad_top, pad_bottom) * 2 + 1)),
    )
    expanded = cv2.dilate((target_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)

    gate = np.zeros_like(target_mask, dtype=np.uint8)
    gate_y1 = max(0, y1 - pad_top)
    gate_y2 = min(height, y2 + pad_bottom + max(1, feather // 8))
    gate_x1 = max(0, x1 - pad_x)
    gate_x2 = min(width, x2 + pad_x)
    gate[gate_y1:gate_y2, gate_x1:gate_x2] = 255

    expanded = cv2.bitwise_and(expanded, gate)
    working_mask = cv2.max(working_mask, expanded)

    if keepout_mask is not None and np.any(keepout_mask):
        keepout = cv2.dilate(
            (keepout_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        working_mask = cv2.bitwise_and(working_mask, cv2.bitwise_not(keepout))

    return working_mask


def _build_lower_nose_cleanup_alpha(target_mask, working_mask, cv2, np, feather: int):
    bbox = compute_mask_bbox(target_mask, np)
    if bbox is None:
        return np.zeros_like(target_mask, dtype=np.float32)

    height, width = target_mask.shape[:2]
    x1, y1, x2, y2 = bbox
    mask_w = max(1, x2 - x1)
    mask_h = max(1, y2 - y1)

    gate = np.zeros_like(target_mask, dtype=np.uint8)
    gate_y1 = max(0, int(round(y1 + mask_h * 0.48)))
    gate_y2 = min(height, y2 + int(round(mask_h * 0.20)))
    gate_x1 = max(0, x1 - int(round(mask_w * 0.20)))
    gate_x2 = min(width, x2 + int(round(mask_w * 0.20)))
    gate[gate_y1:gate_y2, gate_x1:gate_x2] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_at_least(mask_w * 0.26), _odd_at_least(mask_h * 0.16)),
    )
    cleanup = cv2.dilate((target_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
    cleanup = cv2.bitwise_and(cleanup, gate)
    cleanup = cv2.bitwise_and(cleanup, (working_mask > 0).astype(np.uint8) * 255)
    if not np.any(cleanup):
        return np.zeros_like(target_mask, dtype=np.float32)

    core_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_at_least(mask_w * 0.10), _odd_at_least(mask_h * 0.08)),
    )
    core = cv2.erode(cleanup, core_kernel, iterations=1)
    blur_size = _odd_at_least(max(3, feather // 2))
    soft_cleanup = cv2.GaussianBlur(cleanup, (blur_size, blur_size), 0)
    alpha = np.clip(soft_cleanup.astype(np.float32) / 255.0, 0.0, 1.0) * 0.98
    alpha[core > 0] = 1.0
    return alpha


def _build_composite_mask(mask_uint8, cv2, np, feather: int):
    hard = (mask_uint8 > 0).astype(np.uint8) * 255
    if feather <= 0 or not np.any(hard):
        return hard

    blur_size = _odd_at_least(max(3, int(round(feather * 0.45))))
    soft = cv2.GaussianBlur(hard, (blur_size, blur_size), 0)
    core_size = _odd_at_least(max(3, int(round(feather * 0.35))))
    core = cv2.erode(
        hard,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size)),
        iterations=1,
    )
    soft[core > 0] = 255
    return soft.astype(np.uint8)


def _clean_original_region(base_bgr, cleanup_mask, cv2, radius: int):
    hard = (cleanup_mask > 0).astype("uint8") * 255
    if radius <= 0 or not hard.any():
        return base_bgr.copy()
    return cv2.inpaint(base_bgr, hard, int(max(1, radius)), cv2.INPAINT_TELEA)


def warp_masked_region_with_smooth_geometry(
    base_bgr,
    target_mask,
    cv2,
    np,
    scale_x: float = 0.76,
    scale_y: float = 0.94,
    feather: int = 14,
    center_y_ratio: float | None = None,
    lower_extension_ratio: float = 0.0,
    keepout_mask=None,
    clean_background: bool = False,
    background_inpaint_radius: int = 3,
):
    bbox = compute_mask_bbox(target_mask, np)
    if bbox is None:
        return base_bgr, target_mask

    image_h, image_w = target_mask.shape[:2]
    working_mask = _build_working_mask(target_mask, cv2, np, lower_extension_ratio, keepout_mask=keepout_mask)
    working_mask = _expand_working_mask_for_geometry(target_mask, working_mask, cv2, np, scale_x, scale_y, feather, keepout_mask=keepout_mask)
    if not np.any(working_mask):
        return base_bgr, target_mask

    x1, y1, x2, y2 = bbox
    mask_w = max(1, x2 - x1)
    mask_h = max(1, y2 - y1)

    ys, xs = np.where(target_mask > 0)
    center_x = float(xs.mean()) if len(xs) else float(x1 + mask_w * 0.5)
    if center_y_ratio is None:
        center_y_ratio = 0.30 if scale_y > 1.0 else 0.58
    center_y = float(y1 + mask_h * float(center_y_ratio))

    scale_x = float(np.clip(scale_x, 0.55, 1.35))
    scale_y = float(np.clip(scale_y, 0.65, 1.35))
    edge_alpha = _inner_feather_mask(working_mask, feather, cv2, np)
    if scale_y > 1.0:
        cleanup_alpha = _build_lower_nose_cleanup_alpha(target_mask, working_mask, cv2, np, feather)
        edge_alpha = np.maximum(edge_alpha, cleanup_alpha)

    grid_y, grid_x = np.indices((image_h, image_w), dtype=np.float32)
    sigma_x = max(1.0, mask_w * (0.52 if scale_x < 1.0 else 0.46))
    sigma_y = max(1.0, mask_h * (0.72 if scale_y > 1.0 else 0.55))
    focus = np.exp(
        -0.5
        * (
            ((grid_x - center_x) / sigma_x) ** 2
            + ((grid_y - center_y) / sigma_y) ** 2
        )
    ).astype(np.float32)
    if np.any(working_mask > 0):
        focus /= max(float(focus[working_mask > 0].max()), 1e-6)

    weight = np.power(np.clip(edge_alpha * focus, 0.0, 1.0), 0.72)
    source_x = grid_x + weight * (grid_x - center_x) * ((1.0 / scale_x) - 1.0)
    source_y = grid_y + weight * (grid_y - center_y) * ((1.0 / scale_y) - 1.0)
    source_x = np.clip(source_x, 0, image_w - 1).astype(np.float32)
    source_y = np.clip(source_y, 0, image_h - 1).astype(np.float32)

    warped = cv2.remap(
        base_bgr,
        source_x,
        source_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped[working_mask <= 0] = base_bgr[working_mask <= 0]
    if not clean_background:
        change_mask = (edge_alpha * 255.0).clip(0, 255).astype(np.uint8)
        return warped, change_mask

    clean_background_bgr = _clean_original_region(base_bgr, working_mask, cv2, background_inpaint_radius)
    paste_alpha = _inner_feather_mask(
        working_mask,
        max(2, int(round(feather * 0.35))),
        cv2,
        np,
    )[..., None]
    edited = (
        warped.astype(np.float32) * paste_alpha
        + clean_background_bgr.astype(np.float32) * (1.0 - paste_alpha)
    ).clip(0, 255).astype(np.uint8)
    edited[working_mask <= 0] = base_bgr[working_mask <= 0]
    change_mask = _build_composite_mask(working_mask, cv2, np, max(3, int(round(feather * 0.45))))
    return edited, change_mask


def shrink_masked_region_with_smooth_warp(
    base_bgr,
    target_mask,
    cv2,
    np,
    scale_x: float = 0.76,
    scale_y: float = 0.94,
    feather: int = 14,
):
    return warp_masked_region_with_smooth_geometry(
        base_bgr,
        target_mask,
        cv2,
        np,
        scale_x=scale_x,
        scale_y=scale_y,
        feather=feather,
    )


def shrink_masked_region_with_inpaint(
    base_bgr,
    target_mask,
    cv2,
    np,
    scale_x: float = 0.76,
    scale_y: float = 0.94,
    feather: int = 5,
    inpaint_radius: int = 5,
    center_y_ratio: float | None = None,
    lower_extension_ratio: float = 0.0,
    keepout_mask=None,
):
    return warp_masked_region_with_smooth_geometry(
        base_bgr,
        target_mask,
        cv2,
        np,
        scale_x=scale_x,
        scale_y=scale_y,
        feather=max(10, feather * 3),
        center_y_ratio=center_y_ratio,
        lower_extension_ratio=lower_extension_ratio,
        keepout_mask=keepout_mask,
        clean_background=inpaint_radius > 0,
        background_inpaint_radius=inpaint_radius,
    )
