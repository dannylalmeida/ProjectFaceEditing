from __future__ import annotations

import argparse
import json
from pathlib import Path

from face_pipeline_utils import (
    configure_runtime_warnings,
    detect_faces_and_crops,
    ensure_directories,
    resolve_input_image_path,
    suppress_known_stderr_noise,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta o crop principal detetado pelo RetinaFace.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Imagem, pasta ou alias dataset. Se omitido, usa a pasta dataset.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Diretorio onde guardar o crop.")
    parser.add_argument("--margin-scale", type=float, default=0.15, help="Margem extra aplicada a bbox da face.")
    args = parser.parse_args()

    input_path = resolve_input_image_path(args.input)
    output_dir = Path(args.output_dir).resolve()

    configure_runtime_warnings()
    ensure_directories(output_dir=output_dir)

    with suppress_known_stderr_noise():
        _, faces, modules = detect_faces_and_crops(input_path, margin_scale=args.margin_scale)
        cv2 = modules["cv2"]
        primary_face = faces[0]

        crop_path = output_dir / "primary_face_crop.png"
        metadata_path = output_dir / "primary_face.json"

        cv2.imwrite(str(crop_path), primary_face["crop"])
        metadata = {
            "input_image": str(input_path),
            "face_name": primary_face["name"],
            "detected_faces_count": len(faces),
            "selection_strategy": "highest priority, then RetinaFace score; priority combines confidence, face area and centrality",
            "score": primary_face["score"],
            "priority": primary_face.get("priority"),
            "bbox": list(primary_face["crop_bbox"]),
            "detected_bbox": list(primary_face["detected_bbox"]),
            "crop_bbox": list(primary_face["crop_bbox"]),
            "landmarks": primary_face.get("landmarks", {}),
            "alignment": primary_face.get("alignment", {"enabled": False}),
            "crop_path": str(crop_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"Imagem selecionada: {input_path}")
        print(f"Crop principal guardado em: {crop_path}")
        print(f"Metadados guardados em: {metadata_path}")


if __name__ == "__main__":
    main()
