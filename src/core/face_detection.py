from __future__ import annotations

from pathlib import Path

from src.legacy import ensure_legacy_scripts_on_path


def detect_face_retinaface(image_path: str | Path, margin_scale: float = 0.15):
    ensure_legacy_scripts_on_path()
    from face_pipeline_utils import detect_faces_and_crops

    original_bgr, faces, modules = detect_faces_and_crops(Path(image_path), margin_scale=margin_scale)
    if not faces:
        raise ValueError("RetinaFace did not return any valid face crop.")
    return original_bgr, faces[0], modules

