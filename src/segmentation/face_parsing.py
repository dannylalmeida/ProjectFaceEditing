from __future__ import annotations

from src.legacy import ensure_legacy_scripts_on_path
from src.segmentation.mask_utils import build_region_mask, get_region_labels


def load_face_parsing_runtime():
    ensure_legacy_scripts_on_path()
    from face_pipeline_utils import (
        build_parsing_model,
        configure_runtime_warnings,
        load_required_modules,
        suppress_known_stderr_noise,
    )

    configure_runtime_warnings()
    with suppress_known_stderr_noise():
        cv2, np, torch, _, init_parsing_model, img2tensor, normalize = load_required_modules()
        model = build_parsing_model(init_parsing_model)
    return cv2, np, torch, img2tensor, normalize, model


def run_face_parsing(aligned_original_bgr, cv2, np, torch, img2tensor, normalize, model):
    ensure_legacy_scripts_on_path()
    from face_pipeline_utils import parse_face_crop

    return parse_face_crop(aligned_original_bgr, model, cv2, np, torch, img2tensor, normalize)


def build_standard_masks(parsing_map, cv2, np) -> dict[str, object]:
    labels = get_region_labels()
    height, width = parsing_map.shape[:2]

    hair_mask = build_region_mask(parsing_map, labels["cabelo"], 0, cv2, np)
    face_skin_mask = build_region_mask(parsing_map, labels["pele"], 0, cv2, np)
    eyes_mask = build_region_mask(parsing_map, labels["olhos"], 0, cv2, np)
    mouth_mask = build_region_mask(parsing_map, labels["boca"], 0, cv2, np)
    teeth_mask = build_region_mask(parsing_map, [11], 0, cv2, np)
    neck_mask = build_region_mask(parsing_map, labels["pescoco"], 0, cv2, np)
    background_mask = build_region_mask(parsing_map, [0], 0, cv2, np)

    lower_half = np.zeros((height, width), dtype=np.uint8)
    lower_half[int(height * 0.52):height, :] = 255
    lower_face_mask = cv2.bitwise_and(face_skin_mask, lower_half)
    beard_mask = cv2.bitwise_and(lower_face_mask, cv2.bitwise_not(mouth_mask))

    return {
        "hair_mask": hair_mask,
        "face_mask": face_skin_mask,
        "face_skin_mask": face_skin_mask,
        "eyes_mask": eyes_mask,
        "mouth_mask": mouth_mask,
        "teeth_mask": teeth_mask,
        "neck_mask": neck_mask,
        "background_mask": background_mask,
        "lower_face_mask": lower_face_mask,
        "beard_mask": beard_mask,
    }

