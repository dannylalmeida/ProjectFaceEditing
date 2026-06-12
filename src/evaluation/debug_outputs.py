from __future__ import annotations

import json
from pathlib import Path

from src.core.image_io import save_bgr_image, save_rgb_image
from src.segmentation.mask_utils import build_mask_overlay


def compute_diff_stats(before_bgr, after_bgr, mask_uint8, np) -> dict[str, object]:
    diff = np.abs(after_bgr.astype(np.int16) - before_bgr.astype(np.int16))
    max_per_pixel = diff.max(axis=2)
    inside = mask_uint8 > 0
    outside = ~inside

    def stats(region):
        if not np.any(region):
            return {"pixels": 0, "changed_pixels": 0, "max_diff": 0, "mean_diff_changed": 0.0}
        changed = max_per_pixel[region] > 0
        changed_values = diff[region][changed]
        return {
            "pixels": int(region.sum()),
            "changed_pixels": int(changed.sum()),
            "max_diff": int(max_per_pixel[region].max()),
            "mean_diff_changed": float(changed_values.mean()) if changed_values.size else 0.0,
        }

    return {"inside_mask": stats(inside), "outside_mask": stats(outside)}


def build_difference_image(before_bgr, after_bgr, mask_uint8, cv2, np, outside: bool):
    diff = cv2.absdiff(after_bgr, before_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    keep = (mask_uint8 <= 0).astype(np.uint8) if outside else (mask_uint8 > 0).astype(np.uint8)
    diff_gray = (diff_gray * keep).astype(np.uint8)
    return cv2.applyColorMap(diff_gray, cv2.COLORMAP_INFERNO)


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


def save_overlay(path: Path, base_bgr, mask_uint8, cv2, np, color_bgr=(0, 255, 0), alpha: float = 0.24):
    return save_bgr_image(path, build_mask_overlay(base_bgr, mask_uint8, cv2, np, color_bgr=color_bgr, alpha=alpha), cv2)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text_log(path: Path, values: dict[str, object]) -> None:
    lines = [f"{key}: {value}" for key, value in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
