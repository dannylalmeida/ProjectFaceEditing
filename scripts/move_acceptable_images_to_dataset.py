from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_dataset_nose_edit import AuditThresholds, classify_metrics, load_json  # noqa: E402
from src.pipeline.hybrid_pipeline import HybridPipelineConfig, run_hybrid_pipeline  # noqa: E402


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_DESCRIPTION = "nariz maior e mais curto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testa imagens novas e move apenas as aceitaveis para dataset.")
    parser.add_argument("--dataset-dir", default=str(PROJECT_DIR / "dataset"))
    parser.add_argument("--source-dir", default="", help="Processa apenas uma pasta especifica.")
    parser.add_argument("--source-prefix", default="images1024x1024-", help="Prefixo das pastas novas.")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--edit-region", default="nose")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--temp-dir", default=str(PROJECT_DIR / "outputs" / "_tmp_acceptance_test"))
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source_images(source_prefix: str, source_dir: str, dataset_dir: Path, temp_dir: Path) -> list[Path]:
    excluded_roots = {
        dataset_dir.resolve(),
        temp_dir.resolve(),
        (PROJECT_DIR / "outputs").resolve(),
        (PROJECT_DIR / "third_party").resolve(),
    }
    if source_dir:
        roots = [Path(source_dir).resolve()]
    else:
        roots = [
            path
            for path in PROJECT_DIR.iterdir()
            if path.is_dir() and path.name.startswith(source_prefix)
        ]
    images: list[Path] = []
    for root in sorted(roots):
        resolved_root = root.resolve()
        if any(resolved_root == excluded or excluded in resolved_root.parents for excluded in excluded_roots):
            continue
        images.extend(
            sorted(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        )
    return images


def remove_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def run_acceptance_test(image_path: Path, temp_dir: Path, description: str, edit_region: str) -> tuple[bool, str]:
    metadata = run_hybrid_pipeline(
        HybridPipelineConfig(
            input_image=image_path,
            output_dir=temp_dir,
            description=description,
            target_description=description,
            edit_region=edit_region,
            use_face_parsing=True,
            use_local_recolor=False,
            use_styleclip=False,
            use_repaint=False,
            debug=True,
        )
    )
    primary_face_path = temp_dir / "primary_face.json"
    primary_face = load_json(primary_face_path) if primary_face_path.exists() else {}
    classification = classify_metrics(metadata, metadata["validation_report"], primary_face, AuditThresholds())
    grade = classification["grade"]
    return grade in {"bom", "mediano"}, grade


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    temp_dir = Path(args.temp_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    existing_by_name = {
        path.name: path.resolve()
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    images = iter_source_images(args.source_prefix, args.source_dir, dataset_dir, temp_dir)
    if args.limit > 0:
        images = images[: args.limit]

    total = len(images)
    tested = 0
    moved = 0
    rejected = 0
    already_present = 0
    name_conflicts = 0
    failures = 0
    started = time.time()

    print(f"Imagens encontradas nas novas pastas: {total}", flush=True)
    print(f"Dataset destino: {dataset_dir}", flush=True)

    try:
        for index, image_path in enumerate(images, start=1):
            dest = dataset_dir / image_path.name
            existing = existing_by_name.get(image_path.name)
            if existing is not None:
                if file_hash(existing) == file_hash(image_path):
                    image_path.unlink()
                    already_present += 1
                else:
                    name_conflicts += 1
                continue

            tested += 1
            try:
                acceptable, grade = run_acceptance_test(image_path, temp_dir, args.description, args.edit_region)
            except Exception as exc:  # noqa: BLE001 - falha de teste conta como rejeicao.
                failures += 1
                grade = f"falhou: {type(exc).__name__}: {exc}"
                acceptable = False
            finally:
                remove_temp_dir(temp_dir)

            if acceptable:
                if dest.exists():
                    name_conflicts += 1
                else:
                    shutil.move(str(image_path), str(dest))
                    existing_by_name[dest.name] = dest.resolve()
                    moved += 1
            else:
                rejected += 1

            if index % max(1, args.progress_every) == 0 or index == total:
                elapsed = max(1.0, time.time() - started)
                rate = index / elapsed
                remaining = int((total - index) / rate) if rate > 0 else 0
                print(
                    f"[{index}/{total}] testadas={tested} movidas={moved} rejeitadas={rejected} "
                    f"ja_presentes={already_present} conflitos={name_conflicts} falhas={failures} "
                    f"ultimo={image_path.name}:{grade} eta_s={remaining}",
                    flush=True,
                )
    finally:
        remove_temp_dir(temp_dir)

    print("Concluido.", flush=True)
    print(f"Testadas: {tested}", flush=True)
    print(f"Movidas para dataset: {moved}", flush=True)
    print(f"Rejeitadas/maus: {rejected}", flush=True)
    print(f"Ja estavam no dataset e foram removidas da origem: {already_present}", flush=True)
    print(f"Conflitos de nome nao movidos: {name_conflicts}", flush=True)
    print(f"Falhas de pipeline: {failures}", flush=True)


if __name__ == "__main__":
    main()
