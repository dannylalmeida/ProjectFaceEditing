from __future__ import annotations

import argparse
import json
from pathlib import Path

from face_pipeline_utils import (
    REGION_LABELS,
    build_region_mask,
    build_parsing_model,
    configure_runtime_warnings,
    detect_faces_and_crops,
    ensure_directories,
    load_required_modules,
    parse_face_crop,
    suppress_known_stderr_noise,
)


def save_image(path: Path, image_rgb) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def build_keep_mask(parsing_mask, target_labels, dilation: int, cv2, np):
    target_region = build_region_mask(parsing_mask, target_labels, dilation, cv2, np)
    keep_mask = np.where(target_region > 0, 0, 255).astype(np.uint8)
    keep_mask_rgb = cv2.cvtColor(keep_mask, cv2.COLOR_GRAY2RGB)
    target_preview = np.zeros((*keep_mask.shape, 3), dtype=np.uint8)
    target_preview[..., 2] = target_region
    return keep_mask_rgb, target_preview


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara o crop principal, o parsing e a keep mask para o RePaint."
    )
    parser.add_argument("--input", type=str, required=True, help="Imagem de entrada.")
    parser.add_argument("--output-dir", type=str, required=True, help="Diretorio base dos outputs.")
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        choices=sorted(REGION_LABELS.keys()),
        help="Regiao facial a apagar para o RePaint preencher.",
    )
    parser.add_argument("--margin-scale", type=float, default=0.15, help="Margem extra para o crop.")
    parser.add_argument("--dilation", type=int, default=6, help="Expansao extra da mascara alvo.")
    parser.add_argument("--size", type=int, default=256, help="Resolucao final do crop e da mask.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    gt_dir = output_dir / "gts"
    mask_dir = output_dir / "gt_keep_masks"
    debug_dir = output_dir / "debug"

    configure_runtime_warnings()
    ensure_directories(output_dir=output_dir)

    with suppress_known_stderr_noise():
        cv2, np, torch, _, init_parsing_model, img2tensor, normalize = load_required_modules()
        _, faces, _ = detect_faces_and_crops(input_path, margin_scale=args.margin_scale)
        model = build_parsing_model(init_parsing_model)
        primary_face = faces[0]
        crop_bgr = primary_face["crop"]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.resize(crop_rgb, (args.size, args.size), interpolation=cv2.INTER_LINEAR)

        parsing_mask = parse_face_crop(crop_bgr, model, cv2, np, torch, img2tensor, normalize)
        parsing_mask = cv2.resize(parsing_mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

        keep_mask_rgb, target_preview = build_keep_mask(
            parsing_mask=parsing_mask,
            target_labels=REGION_LABELS[args.region],
            dilation=args.dilation,
            cv2=cv2,
            np=np,
        )

    gt_path = gt_dir / "face.png"
    keep_mask_path = mask_dir / "face.png"
    parsing_debug_path = debug_dir / "parsing_labels.png"
    target_preview_path = debug_dir / "target_region.png"
    metadata_path = output_dir / "metadata.json"

    save_image(gt_path, crop_rgb)
    save_image(keep_mask_path, keep_mask_rgb)
    save_image(parsing_debug_path, parsing_mask)
    save_image(target_preview_path, target_preview)

    metadata = {
        "input_image": str(input_path),
        "region": args.region,
        "region_labels": REGION_LABELS[args.region],
        "margin_scale": args.margin_scale,
        "dilation": args.dilation,
        "size": args.size,
        "crop_bbox": list(primary_face["bbox"]),
        "crop_score": primary_face["score"],
        "gt_path": str(gt_path),
        "keep_mask_path": str(keep_mask_path),
        "parsing_debug_path": str(parsing_debug_path),
        "target_preview_path": str(target_preview_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Crop RePaint guardado em: {gt_path}")
    print(f"Keep mask guardada em: {keep_mask_path}")
    print(f"Metadados guardados em: {metadata_path}")


if __name__ == "__main__":
    main()
