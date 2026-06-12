from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from face_pipeline_utils import (
    DATASET_ALIASES,
    OUTPUT_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    configure_runtime_warnings,
    ensure_directories,
    list_dataset_images,
    resolve_input_image_candidates,
    resolve_input_image_path,
    suppress_known_stderr_noise,
)
from src.segmentation.facemesh_region import (
    build_custom_nose_mask,
    detect_facemesh_landmarks,
    draw_numbered_facemesh_overlay,
)
from src.segmentation.mask_utils import build_mask_overlay, extract_mask_outline_contours, normalize_edit_region


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_dataset_images(input_value: str | Path | None) -> list[Path]:
    input_text = str(input_value or "").strip()
    normalized_input = input_text.replace("/", "\\").rstrip("\\")
    if not input_text or normalized_input in DATASET_ALIASES:
        return list_dataset_images()
    else:
        input_path = Path(input_text)
        if input_path.is_dir():
            root = input_path.resolve()
        else:
            return [resolve_input_image_path(input_value)]

    image_paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def build_editable_region_outputs(image_bgr, edit_region: str, cv2, np, landmarks, mp):
    region = normalize_edit_region(edit_region)
    if region == "auto":
        region = "nose"
    if region not in {"nose"}:
        return None

    mask = build_custom_nose_mask(image_bgr, cv2, np, landmarks=landmarks, mp=mp)
    if mask is None:
        return None

    outline_overlay = build_mask_overlay(image_bgr, mask, cv2, np, alpha=0.18)
    contours = extract_mask_outline_contours(mask, cv2, np)
    return {
        "region": region,
        "source": "facemesh_nose",
        "mask": mask,
        "overlay": outline_overlay,
        "contours": contours,
    }


def export_single_image(input_path: Path, output_dir: Path, display_scale: float, edit_region: str) -> dict[str, object]:
    import cv2
    import numpy as np

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {input_path}")

    landmarks, mp = detect_facemesh_landmarks(image_bgr, cv2)
    if landmarks is None:
        raise ValueError(f"Nenhuma Face Mesh foi detetada na imagem: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    landmarks_path = output_dir / "landmarks_478_enumerados.png"
    overlay = draw_numbered_facemesh_overlay(
        image_bgr,
        cv2,
        scale=max(1.0, display_scale),
        landmarks=landmarks,
        mp=mp,
    )
    cv2.imwrite(str(landmarks_path), overlay)

    result: dict[str, object] = {
        "input_image": str(input_path),
        "landmarks_path": str(landmarks_path),
        "landmarks_count": len(landmarks),
    }
    editable_region = build_editable_region_outputs(image_bgr, edit_region, cv2, np, landmarks, mp)
    if editable_region is not None:
        outline_path = output_dir / "editable_region_outline.png"
        metadata_path = output_dir / "editable_region_outline.json"
        cv2.imwrite(str(outline_path), editable_region["overlay"])
        payload = {
            "input_image": str(input_path),
            "edit_region": editable_region["region"],
            "source": editable_region["source"],
            "coordinate_space": "image",
            "contours": editable_region["contours"],
            "mask_pixels": int((editable_region["mask"] > 0).sum()),
            "outline_path": str(outline_path),
            "landmarks_path": str(landmarks_path),
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result.update(
            {
                "editable_region": editable_region["region"],
                "editable_region_outline_path": str(outline_path),
                "editable_region_metadata_path": str(metadata_path),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera Face Mesh numerada e delineado automatico da regiao editavel.")
    parser.add_argument("--input", type=str, default="", help="Imagem/pasta de entrada. Se omitido, usa dataset.")
    parser.add_argument("--output-dir", type=str, default="", help="Diretorio de saida.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Mantido por compatibilidade.")
    parser.add_argument("--display-scale", type=float, default=2.0, help="Escala visual dos numeros.")
    parser.add_argument("--edit-region", type=str, default="nose", help="Regiao editavel a delinear. Usa none para so landmarks.")
    parser.add_argument("--batch", action="store_true", help="Processa todas as imagens da pasta de entrada.")
    args = parser.parse_args()

    configure_runtime_warnings()
    if args.batch:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR / "landmarks_dataset"
        input_paths = iter_dataset_images(args.input)
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR / "landmarks"
        input_paths = resolve_input_image_candidates(args.input, max_attempts=25)

    ensure_directories(output_dir=output_dir)
    reset_output_dir(output_dir)
    with suppress_known_stderr_noise():
        results = []
        failures = []
        for index, input_path in enumerate(input_paths, start=1):
            target_dir = output_dir / f"{index:04d}_{input_path.stem}" if args.batch else output_dir
            try:
                results.append(export_single_image(input_path, target_dir, args.display_scale, args.edit_region))
                if not args.batch:
                    break
            except Exception as exc:
                failures.append(f"{input_path}: {type(exc).__name__}: {exc}")
                if args.batch:
                    continue
        if not results:
            details = "\n".join(failures[:8])
            raise ValueError(
                "Nao consegui encontrar uma imagem valida para gerar landmarks nas tentativas aleatorias.\n"
                f"Tentativas falhadas:\n{details}"
            )
        if args.batch and failures:
            (output_dir / "batch_failures.json").write_text(
                json.dumps({"failures": failures}, indent=2),
                encoding="utf-8",
            )

    print("")
    print("FACE MESH OK")
    print(f"Imagens processadas: {len(results)}")
    print(f"Diretorio de output: {output_dir}")
    if not args.batch and results:
        result = results[0]
        print(f"Imagem original: {result['input_image']}")
        print("Landmarks enumerados: 0..477")
        print(f"Landmarks: {result['landmarks_path']}")
        if result.get("editable_region_outline_path"):
            print(f"Delineado automatico: {result['editable_region_outline_path']}")
            print(f"Metadados do delineado: {result['editable_region_metadata_path']}")


if __name__ == "__main__":
    main()
