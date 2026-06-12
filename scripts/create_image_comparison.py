from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def load_rgb(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image


def make_comparison(left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
    height = max(left.height, right.height)
    if left.height != height:
        left = left.resize((round(left.width * height / left.height), height), Image.Resampling.LANCZOS)
    if right.height != height:
        right = right.resize((round(right.width * height / right.height), height), Image.Resampling.LANCZOS)

    header_height = 30
    divider_width = 2
    output = Image.new("RGB", (left.width + divider_width + right.width, height + header_height), "white")
    draw = ImageDraw.Draw(output)
    output.paste(left, (0, header_height))
    output.paste(right, (left.width + divider_width, header_height))
    draw.rectangle((left.width, header_height, left.width + divider_width - 1, height + header_height), fill=(0, 0, 0))
    draw.text((8, 8), left_label, fill=(0, 0, 0))
    draw.text((left.width + divider_width + 8, 8), right_label, fill=(0, 0, 0))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria uma comparacao lado a lado entre duas imagens.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--left-label", default="original")
    parser.add_argument("--right-label", default="resultado")
    args = parser.parse_args()

    left = load_rgb(Path(args.left))
    right = load_rgb(Path(args.right))
    output = make_comparison(left, right, args.left_label, args.right_label)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    print(f"Comparacao guardada em: {output_path}")


if __name__ == "__main__":
    main()
