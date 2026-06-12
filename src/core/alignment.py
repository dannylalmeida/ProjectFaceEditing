from __future__ import annotations

from pathlib import Path

from src.core.face_detection import detect_face_retinaface


def align_and_crop_face(image_path: str | Path, margin_scale: float = 0.15):
    original_bgr, primary_face, modules = detect_face_retinaface(image_path, margin_scale=margin_scale)
    return {
        "original_bgr": original_bgr,
        "aligned_bgr": primary_face["crop"],
        "bbox": primary_face.get("crop_bbox") or primary_face.get("bbox"),
        "landmarks": primary_face.get("landmarks", {}),
        "alignment": primary_face.get("alignment", {"enabled": False}),
        "score": primary_face.get("score"),
        "priority": primary_face.get("priority"),
        "modules": modules,
    }


def draw_landmarks_overlay(image_bgr, landmarks: dict[str, tuple[float, float]], cv2):
    overlay = image_bgr.copy()
    for name, point in (landmarks or {}).items():
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(overlay, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return overlay

