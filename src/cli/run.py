from __future__ import annotations

import argparse
import json

from src.pipeline.hybrid_pipeline import HybridPipelineConfig, run_hybrid_pipeline


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "sim", "s", "on"}:
        return True
    if normalized in {"0", "false", "no", "nao", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid face edit pipeline.")
    parser.add_argument("--input", default="novo_dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-metadata", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--source-description", default="")
    parser.add_argument("--target-description", default="")
    parser.add_argument("--edit-region", default="auto")
    parser.add_argument("--use-face-parsing", type=parse_bool, default=True)
    parser.add_argument("--use-local-recolor", type=parse_bool, default=True)
    parser.add_argument("--use-styleclip", type=parse_bool, default=False)
    parser.add_argument("--styleclip-edited-image", default="")
    parser.add_argument("--mask-dilation", type=int, default=-1)
    parser.add_argument("--mask-erosion", type=int, default=0)
    parser.add_argument("--mask-blur", type=int, default=-1)
    parser.add_argument("--mask-threshold", type=int, default=1)
    parser.add_argument("--use-repaint", type=parse_bool, default=False)
    parser.add_argument("--repaint-steps", type=int, default=20)
    parser.add_argument("--repaint-strength", type=float, default=0.35)
    parser.add_argument("--repaint-backend", choices=["opencv", "repaint"], default="opencv")
    parser.add_argument("--debug", type=parse_bool, default=False)
    parser.add_argument("--margin-scale", type=float, default=0.15)
    parser.add_argument("--local-strength", type=float, default=0.82)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = HybridPipelineConfig(
        input_image=args.input,
        output_dir=args.output_dir,
        crop_metadata=args.crop_metadata or None,
        description=args.description,
        source_description=args.source_description,
        target_description=args.target_description,
        edit_region=args.edit_region,
        use_face_parsing=args.use_face_parsing,
        use_local_recolor=args.use_local_recolor,
        use_styleclip=args.use_styleclip,
        styleclip_edited_image=args.styleclip_edited_image or None,
        mask_dilation=args.mask_dilation,
        mask_erosion=args.mask_erosion,
        mask_blur=args.mask_blur,
        mask_threshold=args.mask_threshold,
        use_repaint=args.use_repaint,
        repaint_steps=args.repaint_steps,
        repaint_strength=args.repaint_strength,
        repaint_backend=args.repaint_backend,
        debug=args.debug,
        margin_scale=args.margin_scale,
        local_strength=args.local_strength,
    )
    metadata = run_hybrid_pipeline(config)
    print(json.dumps({"ok": True, "metadata": metadata["paths"]}, indent=2))


if __name__ == "__main__":
    main()
