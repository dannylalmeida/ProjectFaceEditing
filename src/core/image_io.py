from __future__ import annotations

from pathlib import Path


def load_image(path: str | Path, cv2):
    image_path = Path(path).resolve()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image_bgr


def save_rgb_image(path: str | Path, image_rgb) -> Path:
    from PIL import Image

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(output_path)
    return output_path


def save_bgr_image(path: str | Path, image_bgr, cv2) -> Path:
    return save_rgb_image(path, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def validate_rgb_uint8(name: str, image, np) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be an RGB/BGR image with 3 channels; got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8; got {image.dtype}")
    if image.min() < 0 or image.max() > 255:
        raise ValueError(f"{name} must be in range 0-255")

