from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def load_rgb(path: Path, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.height != height:
        image = image.resize((round(image.width * height / image.height), height), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria uma comparacao horizontal para sweep StyleCLIP.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", action="append", required=True, help="Entrada no formato label=path")
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    items: list[tuple[str, Path, Image.Image]] = []
    for entry in args.image:
        if "=" not in entry:
            raise ValueError(f"Entrada invalida: {entry}. Usa label=path")
        label, path_text = entry.split("=", 1)
        path = Path(path_text)
        items.append((label, path, load_rgb(path, args.height)))

    header_height = 30
    divider_width = 2
    width = sum(image.width for _, _, image in items) + divider_width * (len(items) - 1)
    output = Image.new("RGB", (width, args.height + header_height), "white")
    draw = ImageDraw.Draw(output)

    x = 0
    for index, (label, _, image) in enumerate(items):
        output.paste(image, (x, header_height))
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
        x += image.width
        if index < len(items) - 1:
            draw.rectangle((x, header_height, x + divider_width - 1, args.height + header_height), fill=(0, 0, 0))
            x += divider_width

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    print(f"Comparacao sweep guardada em: {output_path}")


if __name__ == "__main__":
    main()
