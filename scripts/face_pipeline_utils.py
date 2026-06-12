from __future__ import annotations

import io
import math
import os
import random
import re
import sys
import warnings
from contextlib import contextmanager, redirect_stderr
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "novo_dataset"
EXTRA_INPUT_DIRS = (PROJECT_DIR / "38000",)
DEFAULT_IMAGE_PATH = PROJECT_DIR / "inputs" / "image.png"
MODEL_CACHE_DIR = PROJECT_DIR / "models" / "facexlib"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PARSER_INPUT_SIZE = 512
ALIGNED_FACE_SIZE = 512
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATASET_ALIASES = {
    "dataset",
    ".\\dataset",
    "novo_dataset",
    ".\\novo_dataset",
}
REGION_LABELS = {
    "pele": [1],
    "sobrancelhas": [2, 3],
    "olhos": [4, 5, 6],
    "orelhas": [7, 8, 9],
    "nariz": [10],
    "boca": [11, 12, 13],
    "pescoco": [14, 15],
    "cabelo": [17, 18],
}
STDERR_NOISE_PATTERNS = (
    re.compile(r".*tf\.losses\.sparse_softmax_cross_entropy.*"),
    re.compile(r".*Created TensorFlow Lite XNNPACK delegate.*"),
    re.compile(r".*All log messages before absl::InitializeLog\(\).*"),
    re.compile(r".*inference_feedback_manager\.cc.*"),
    re.compile(r".*landmark_projection_calculator\.cc.*"),
)
IMAGE_LEFT_ALIGNMENT_TEMPLATE = (
    (0.375, 0.46),
    (0.625, 0.46),
    (0.50, 0.61),
    (0.41, 0.77),
    (0.59, 0.77),
)
ANATOMICAL_LEFT_ALIGNMENT_TEMPLATE = (
    (0.625, 0.46),
    (0.375, 0.46),
    (0.50, 0.61),
    (0.59, 0.77),
    (0.41, 0.77),
)


class _FilteredStderr(io.TextIOBase):
    def __init__(self, target, patterns: tuple[re.Pattern[str], ...]) -> None:
        self._target = target
        self._patterns = patterns
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        self._target.flush()

    def _write_line(self, line: str) -> None:
        if any(pattern.search(line) for pattern in self._patterns):
            return
        self._target.write(line)


@contextmanager
def suppress_known_stderr_noise():
    filtered_stderr = _FilteredStderr(sys.stderr, STDERR_NOISE_PATTERNS)
    with redirect_stderr(filtered_stderr):
        yield
    filtered_stderr.flush()


def configure_runtime_warnings() -> None:
    matplotlib_cache_dir = PROJECT_DIR / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    warnings.filterwarnings(
        "ignore",
        message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*tf\.losses\.sparse_softmax_cross_entropy.*",
        category=Warning,
    )


def ensure_directories(output_dir: Path | None = None) -> None:
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    (output_dir or OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def list_dataset_images(dataset_dir: Path | None = None) -> list[Path]:
    if dataset_dir is not None:
        dataset_dirs = (Path(dataset_dir).resolve(),)
    else:
        dataset_dirs = tuple(path.resolve() for path in (DATASET_DIR, *EXTRA_INPUT_DIRS) if path.exists())

    if not dataset_dirs:
        raise FileNotFoundError(f"Pastas de imagens nao encontradas: {DATASET_DIR}, {', '.join(str(path) for path in EXTRA_INPUT_DIRS)}")

    image_files = []
    for resolved_dataset_dir in dataset_dirs:
        if not resolved_dataset_dir.exists():
            raise FileNotFoundError(f"Pasta de imagens nao encontrada: {resolved_dataset_dir}")
        image_files.extend(
            path
            for path in resolved_dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    if not image_files:
        roots = ", ".join(str(path) for path in dataset_dirs)
        raise FileNotFoundError(f"Nao encontrei imagens dentro de: {roots}")

    return sorted(image_files)


def get_random_dataset_image(dataset_dir: Path | None = None) -> Path:
    image_files = list_dataset_images(dataset_dir)
    return random.choice(image_files)


def resolve_input_image_candidates(input_value: str | Path | None = None, max_attempts: int = 25) -> list[Path]:
    input_text = str(input_value or "").strip()
    normalized_input = input_text.replace("/", "\\").rstrip("\\")
    if not input_text or normalized_input in DATASET_ALIASES:
        image_files = list_dataset_images()
        random.shuffle(image_files)
        return image_files[: max(1, min(max_attempts, len(image_files)))]

    input_path = Path(input_text)
    if input_path.is_dir():
        image_files = list_dataset_images(input_path)
        random.shuffle(image_files)
        return image_files[: max(1, min(max_attempts, len(image_files)))]

    return [input_path.resolve()]


def resolve_input_image_path(input_value: str | Path | None = None) -> Path:
    return resolve_input_image_candidates(input_value, max_attempts=1)[0]


def load_required_modules():
    import absl.logging
    import cv2
    import numpy as np
    import tensorflow as tf
    import torch
    from retinaface import RetinaFace
    from facexlib.parsing import init_parsing_model
    from facexlib.utils.misc import img2tensor
    from torchvision.transforms.functional import normalize

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.get_logger().setLevel("ERROR")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    return cv2, np, torch, RetinaFace, init_parsing_model, img2tensor, normalize


def expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    scale: float = 0.15,
) -> tuple[int, int, int, int]:
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * scale)
    pad_y = int(box_h * scale)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return x1, y1, x2, y2


def parse_faces(raw_faces: dict) -> list[tuple[str, dict]]:
    if not isinstance(raw_faces, dict) or not raw_faces:
        return []
    return sorted(raw_faces.items(), key=lambda item: float(item[1]["score"]), reverse=True)


def extract_landmarks(face_data: dict) -> dict[str, tuple[float, float]]:
    landmarks = face_data.get("landmarks") or {}
    required = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
    extracted = {}
    for key in required:
        point = landmarks.get(key)
        if not point or len(point) != 2:
            return {}
        extracted[key] = (float(point[0]), float(point[1]))
    return extracted


def build_alignment_template(size: int, np, landmarks: dict[str, tuple[float, float]]):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    template = ANATOMICAL_LEFT_ALIGNMENT_TEMPLATE if left_eye[0] > right_eye[0] else IMAGE_LEFT_ALIGNMENT_TEMPLATE
    return np.array(
        [(x * size, y * size) for x, y in template],
        dtype=np.float32,
    )


def estimate_alignment(image_bgr, landmarks: dict[str, tuple[float, float]], output_size: int, cv2, np):
    if len(landmarks) < 5:
        return None

    source_points = np.array(
        [
            landmarks["left_eye"],
            landmarks["right_eye"],
            landmarks["nose"],
            landmarks["mouth_left"],
            landmarks["mouth_right"],
        ],
        dtype=np.float32,
    )
    target_points = build_alignment_template(output_size, np, landmarks)
    matrix, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.LMEDS)
    if matrix is None:
        return None

    aligned_crop = cv2.warpAffine(
        image_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    inverse_matrix = cv2.invertAffineTransform(matrix)
    return {
        "crop": aligned_crop,
        "matrix": matrix,
        "inverse_matrix": inverse_matrix,
        "size": output_size,
    }


def compute_face_priority(face_bbox, score: float, image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = face_bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    area_ratio = float(box_w * box_h) / float(max(1, image_width * image_height))

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    norm_dx = (center_x - image_center_x) / max(1.0, image_center_x)
    norm_dy = (center_y - image_center_y) / max(1.0, image_center_y)
    center_distance = math.sqrt(norm_dx * norm_dx + norm_dy * norm_dy)
    centrality = max(0.0, 1.0 - min(center_distance, 1.5) / 1.5)

    area_score = min(area_ratio * 6.0, 1.0)
    return score * 0.55 + area_score * 0.30 + centrality * 0.15


def prepare_face_tensor(face_bgr, cv2, np, torch, img2tensor, normalize):
    face_resized = cv2.resize(face_bgr, (PARSER_INPUT_SIZE, PARSER_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    face_tensor = img2tensor(face_resized.astype(np.float32) / 255.0, bgr2rgb=True, float32=True)
    normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return torch.unsqueeze(face_tensor, 0)


def build_parsing_model(init_parsing_model):
    return init_parsing_model(model_name="bisenet", device="cpu", model_rootpath=str(MODEL_CACHE_DIR))


def parse_face_crop(face_bgr, model, cv2, np, torch, img2tensor, normalize):
    face_tensor = prepare_face_tensor(face_bgr, cv2, np, torch, img2tensor, normalize)
    with torch.no_grad():
        output = model(face_tensor)[0]

    mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    crop_h, crop_w = face_bgr.shape[:2]
    mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    return mask


def build_region_mask(parsing_mask, target_labels, dilation: int, cv2, np):
    region_mask = np.isin(parsing_mask, target_labels).astype(np.uint8) * 255
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        region_mask = cv2.dilate(region_mask, kernel, iterations=1)
    return region_mask


def detect_faces_and_crops(
    image_path: Path,
    margin_scale: float = 0.15,
) -> tuple[object, list[dict], dict]:
    cv2, np, _, RetinaFace, _, _, _ = load_required_modules()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    detections = RetinaFace.detect_faces(str(image_path))
    faces = parse_faces(detections)
    if not faces:
        raise ValueError("Nenhuma face foi detetada pelo RetinaFace.")

    image_height, image_width = image.shape[:2]
    face_results = []
    for index, (face_name, face_data) in enumerate(faces, start=1):
        raw_x1, raw_y1, raw_x2, raw_y2 = [int(value) for value in face_data["facial_area"]]
        crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
            raw_x1,
            raw_y1,
            raw_x2,
            raw_y2,
            image_width,
            image_height,
            scale=margin_scale,
        )
        detected_bbox = (raw_x1, raw_y1, raw_x2, raw_y2)
        crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
        raw_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if raw_crop.size == 0:
            continue

        landmarks = extract_landmarks(face_data)
        alignment = estimate_alignment(image, landmarks, ALIGNED_FACE_SIZE, cv2, np) if landmarks else None
        crop = alignment["crop"] if alignment else raw_crop
        priority = compute_face_priority(detected_bbox, float(face_data["score"]), image_width, image_height)

        face_results.append(
            {
                "index": index,
                "name": face_name,
                "score": float(face_data["score"]),
                "bbox": crop_bbox,
                "detected_bbox": detected_bbox,
                "crop_bbox": crop_bbox,
                "priority": priority,
                "landmarks": landmarks,
                "raw_crop": raw_crop,
                "crop": crop,
                "alignment": {
                    "enabled": alignment is not None,
                    "size": ALIGNED_FACE_SIZE if alignment else None,
                    "matrix": alignment["matrix"].tolist() if alignment else None,
                    "inverse_matrix": alignment["inverse_matrix"].tolist() if alignment else None,
                },
            }
        )

    if not face_results:
        raise ValueError("Foram detetadas faces, mas nenhum crop valido foi produzido.")

    face_results.sort(key=lambda face: (face["priority"], face["score"]), reverse=True)
    for new_index, face in enumerate(face_results, start=1):
        face["index"] = new_index

    return image, face_results, {"cv2": cv2}


def save_face_outputs(face_name: str, crop_bgr, mask_gray, mask_color, overlay_bgr, cv2, output_dir: Path | None = None) -> None:
    target_dir = output_dir or OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_dir / f"{face_name}_crop.png"), crop_bgr)
    cv2.imwrite(str(target_dir / f"{face_name}_mask_gray.png"), mask_gray)
    cv2.imwrite(str(target_dir / f"{face_name}_mask_color.png"), mask_color)
    cv2.imwrite(str(target_dir / f"{face_name}_overlay.png"), overlay_bgr)
