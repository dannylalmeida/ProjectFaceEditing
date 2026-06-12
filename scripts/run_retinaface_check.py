from __future__ import annotations

import argparse
import json
from pathlib import Path

from face_pipeline_utils import (
    OUTPUT_DIR,
    compute_face_priority,
    configure_runtime_warnings,
    ensure_directories,
    extract_landmarks,
    parse_faces,
    resolve_input_image_path,
    suppress_known_stderr_noise,
)


LANDMARK_ORDER = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
LANDMARK_LABELS = {
    "left_eye": "olho_esquerdo",
    "right_eye": "olho_direito",
    "nose": "nariz",
    "mouth_left": "canto_boca_esquerdo",
    "mouth_right": "canto_boca_direito",
}
LANDMARK_COLORS = {
    "left_eye": (255, 180, 0),
    "right_eye": (255, 180, 0),
    "nose": (0, 220, 255),
    "mouth_left": (0, 180, 80),
    "mouth_right": (0, 180, 80),
}


def load_retinaface_modules():
    import absl.logging
    import cv2
    import tensorflow as tf
    from retinaface import RetinaFace

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.get_logger().setLevel("ERROR")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    return cv2, RetinaFace


def point_to_dict(point: tuple[float, float]) -> dict[str, float]:
    return {
        "x": round(float(point[0]), 2),
        "y": round(float(point[1]), 2),
    }


def format_point(point: tuple[float, float]) -> str:
    return f"x={point[0]:.1f}, y={point[1]:.1f}"


def build_detection_result(index: int, face_name: str, face_data: dict, image_width: int, image_height: int) -> dict:
    x1, y1, x2, y2 = [int(value) for value in face_data["facial_area"]]
    landmarks = extract_landmarks(face_data)
    missing_landmarks = [key for key in LANDMARK_ORDER if key not in landmarks]
    score = float(face_data["score"])
    bbox = (x1, y1, x2, y2)
    return {
        "index": index,
        "name": face_name,
        "score": score,
        "priority": compute_face_priority(bbox, score, image_width, image_height),
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
        },
        "landmarks": {
            LANDMARK_LABELS[key]: point_to_dict(landmarks[key])
            for key in LANDMARK_ORDER
            if key in landmarks
        },
        "raw_landmark_names": {
            key: point_to_dict(landmarks[key])
            for key in LANDMARK_ORDER
            if key in landmarks
        },
        "missing_landmarks": missing_landmarks,
        "landmarks_complete": not missing_landmarks,
    }


def draw_detection_overlay(image_bgr, detections: list[dict], cv2):
    overlay = image_bgr.copy()
    for detection in detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(
            overlay,
            f"face {detection['index']} score={detection['score']:.3f}",
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        raw_landmarks = detection["raw_landmark_names"]
        for key, point in raw_landmarks.items():
            x, y = int(round(point["x"])), int(round(point["y"]))
            color = LANDMARK_COLORS[key]
            cv2.circle(overlay, (x, y), 6, color, -1)
            cv2.circle(overlay, (x, y), 9, (0, 0, 0), 2)
            cv2.putText(
                overlay,
                LANDMARK_LABELS[key],
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        if "left_eye" in raw_landmarks and "right_eye" in raw_landmarks:
            left_eye = raw_landmarks["left_eye"]
            right_eye = raw_landmarks["right_eye"]
            cv2.line(
                overlay,
                (int(round(left_eye["x"])), int(round(left_eye["y"]))),
                (int(round(right_eye["x"])), int(round(right_eye["y"]))),
                (255, 180, 0),
                2,
            )
        if "mouth_left" in raw_landmarks and "mouth_right" in raw_landmarks:
            mouth_left = raw_landmarks["mouth_left"]
            mouth_right = raw_landmarks["mouth_right"]
            cv2.line(
                overlay,
                (int(round(mouth_left["x"])), int(round(mouth_left["y"]))),
                (int(round(mouth_right["x"])), int(round(mouth_right["y"]))),
                (0, 180, 80),
                2,
            )

    return overlay


def print_detection_summary(input_path: Path, detections: list[dict], output_dir: Path, overlay_path: Path, metadata_path: Path) -> None:
    print("")
    print("RETINAFACE OK")
    print(f"Imagem analisada: {input_path}")
    print(f"Faces detetadas: {len(detections)}")

    for detection in detections:
        bbox = detection["bbox"]
        print("")
        print(f"Face {detection['index']} ({detection['name']})")
        print(f"  score: {detection['score']:.4f}")
        print(
            "  bbox rosto: "
            f"x1={bbox['x1']}, y1={bbox['y1']}, x2={bbox['x2']}, y2={bbox['y2']}, "
            f"largura={bbox['width']}, altura={bbox['height']}"
        )

        raw = detection["raw_landmark_names"]
        if detection["landmarks_complete"]:
            print(f"  olhos: esquerdo({format_point_tuple(raw['left_eye'])}) | direito({format_point_tuple(raw['right_eye'])})")
            print(f"  nariz: centro({format_point_tuple(raw['nose'])})")
            print(
                "  boca: "
                f"canto esquerdo({format_point_tuple(raw['mouth_left'])}) | "
                f"canto direito({format_point_tuple(raw['mouth_right'])})"
            )
        else:
            print("  landmarks incompletos")
            print("  em falta: " + ", ".join(detection["missing_landmarks"]))

    primary = detections[0]
    print("")
    if primary["landmarks_complete"]:
        print("Confirmacao: RetinaFace detetou a face principal e os 5 pontos principais.")
        print("Pontos confirmados: olho esquerdo, olho direito, centro do nariz, canto esquerdo da boca e canto direito da boca.")
    else:
        print("Confirmacao incompleta: RetinaFace detetou a face, mas faltam landmarks na face principal.")

    print(f"Overlay visual: {overlay_path}")
    print(f"Metadados: {metadata_path}")
    print(f"Pasta de output: {output_dir}")


def format_point_tuple(point_dict: dict[str, float]) -> str:
    return f"x={point_dict['x']:.1f}, y={point_dict['y']:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida apenas o RetinaFace: face, bbox e landmarks de olhos/nariz/boca."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Imagem de entrada. Se omitido, usa imagens de novo_dataset e 38000.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Diretorio onde guardar o overlay e os metadados.",
    )
    args = parser.parse_args()

    input_path = resolve_input_image_path(args.input)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR / "retinaface_check"
    output_dir = output_dir.resolve()

    configure_runtime_warnings()
    ensure_directories(output_dir=output_dir)

    with suppress_known_stderr_noise():
        cv2, RetinaFace = load_retinaface_modules()
        image_bgr = cv2.imread(str(input_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Imagem nao encontrada: {input_path}")

        raw_faces = RetinaFace.detect_faces(str(input_path))
        faces = parse_faces(raw_faces)
        if not faces:
            print("RETINAFACE FALHOU")
            print(f"Imagem analisada: {input_path}")
            print("Nenhuma face foi detetada.")
            raise SystemExit(1)

        image_height, image_width = image_bgr.shape[:2]
        detections = [
            build_detection_result(index, face_name, face_data, image_width, image_height)
            for index, (face_name, face_data) in enumerate(faces, start=1)
        ]
        detections.sort(key=lambda item: (item["priority"], item["score"]), reverse=True)
        for index, detection in enumerate(detections, start=1):
            detection["index"] = index

        overlay = draw_detection_overlay(image_bgr, detections, cv2)
        overlay_path = output_dir / "retinaface_landmarks.png"
        metadata_path = output_dir / "retinaface_detections.json"
        cv2.imwrite(str(overlay_path), overlay)
        metadata_path.write_text(
            json.dumps(
                {
                    "input_image": str(input_path),
                    "faces_detected": len(detections),
                    "primary_face_complete": detections[0]["landmarks_complete"],
                    "detections": detections,
                    "overlay_path": str(overlay_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print_detection_summary(input_path, detections, output_dir, overlay_path, metadata_path)

    if not detections[0]["landmarks_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
