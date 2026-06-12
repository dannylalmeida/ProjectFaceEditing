from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from face_pipeline_utils import list_dataset_images  # noqa: E402
from src.pipeline.hybrid_pipeline import HybridPipelineConfig, run_hybrid_pipeline  # noqa: E402


DEFAULT_DESCRIPTION = "nariz maior e mais curto"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "dataset_audit_nose_short_wide"


@dataclass
class AuditThresholds:
    good_mask_area_min: float = 0.020
    good_mask_area_max: float = 0.075
    bad_mask_area_min: float = 0.012
    bad_mask_area_max: float = 0.110
    good_changed_ratio_min: float = 0.65
    bad_changed_ratio_min: float = 0.35
    good_mean_diff_min: float = 4.0
    good_mean_diff_max: float = 18.0
    bad_mean_diff_min: float = 2.0
    bad_mean_diff_max: float = 35.0
    good_max_diff_max: int = 140
    bad_max_diff_max: int = 200
    good_detection_score_min: float = 0.97
    bad_detection_score_min: float = 0.90
    good_face_area_min: float = 0.035
    bad_face_area_min: float = 0.015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita a edicao local de nariz em todas as imagens do dataset."
    )
    parser.add_argument("--input-dir", default="", help="Pasta de imagens. Vazio usa novo_dataset + 38000.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Pasta raiz do relatorio.")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--edit-region", default="nose")
    parser.add_argument("--limit", type=int, default=0, help="Limita o numero de imagens, para smoke test.")
    parser.add_argument("--resume", action="store_true", help="Reutiliza resultados ja existentes.")
    parser.add_argument("--clean", action="store_true", help="Apaga a pasta de output antes de correr.")
    parser.add_argument("--contact-sheet-width", type=int, default=4)
    parser.add_argument("--review-overrides", default="", help="JSON com downgrades de revisao visual.")
    return parser.parse_args()


def slug_for_image(index: int, image_path: Path) -> str:
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in image_path.stem)
    return f"{index:04d}_{safe_stem}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def percent(value: float) -> float:
    return round(value * 100.0, 3)


def bbox_area_ratio(primary_face: dict[str, Any]) -> float:
    bbox = primary_face.get("detected_bbox") or primary_face.get("bbox")
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, (x2 - x1) * (y2 - y1)) / float(512 * 512)


def classify_metrics(
    metadata: dict[str, Any],
    validation: dict[str, Any],
    primary_face: dict[str, Any],
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    hard: list[str] = []
    warnings: list[str] = []

    strategy = metadata.get("strategy") or {}
    direct_refinement = metadata.get("direct_refinement") or {}
    editable_outline = metadata.get("editable_region_outline") or {}
    alignment = metadata.get("alignment") or {}

    aligned_diff = validation.get("aligned_diff") or {}
    full_diff = validation.get("full_image_diff") or {}
    inside = aligned_diff.get("inside_mask") or {}
    outside_aligned = aligned_diff.get("outside_mask") or {}
    outside_full = full_diff.get("outside_mask") or {}

    mask_pixels_aligned = int(validation.get("mask_pixels_aligned") or 0)
    mask_area_ratio = mask_pixels_aligned / float(512 * 512)
    changed_pixels = int(inside.get("changed_pixels") or 0)
    changed_ratio = changed_pixels / float(max(mask_pixels_aligned, 1))
    mean_diff = float(inside.get("mean_diff_changed") or 0.0)
    max_diff = int(inside.get("max_diff") or 0)
    outside_changed = int(outside_aligned.get("changed_pixels") or 0) + int(outside_full.get("changed_pixels") or 0)
    contour_count = int(validation.get("editable_region_outline_contours") or 0)
    detection_score = float(primary_face.get("score") or 0.0)
    face_ratio = bbox_area_ratio(primary_face)

    if strategy.get("edit_region") != "nose":
        hard.append("regiao editada nao e nariz")
    if direct_refinement.get("mode") != "direct_nose_smooth_warp":
        hard.append("edicao geometrica do nariz nao foi aplicada")
    if editable_outline.get("source") != "facemesh_nose":
        hard.append("mascara do nariz nao veio de FaceMesh")
    if not alignment.get("enabled"):
        hard.append("alinhamento facial nao foi ativado")
    if outside_changed != 0:
        hard.append(f"alterou pixels fora da mascara ({outside_changed})")
    if contour_count < 1:
        hard.append("sem contorno editavel")
    elif contour_count > 1:
        warnings.append(f"mais de um contorno editavel ({contour_count})")

    if mask_area_ratio < thresholds.bad_mask_area_min or mask_area_ratio > thresholds.bad_mask_area_max:
        hard.append(f"area da mascara fora do limite ({percent(mask_area_ratio)}%)")
    elif mask_area_ratio < thresholds.good_mask_area_min or mask_area_ratio > thresholds.good_mask_area_max:
        warnings.append(f"area da mascara suspeita ({percent(mask_area_ratio)}%)")

    if changed_ratio < thresholds.bad_changed_ratio_min:
        hard.append(f"alteracao fraca dentro da mascara ({percent(changed_ratio)}%)")
    elif changed_ratio < thresholds.good_changed_ratio_min:
        warnings.append(f"alteracao moderada dentro da mascara ({percent(changed_ratio)}%)")

    if mean_diff < thresholds.bad_mean_diff_min or mean_diff > thresholds.bad_mean_diff_max:
        hard.append(f"intensidade media alterada fora do aceitavel ({mean_diff:.2f})")
    elif mean_diff < thresholds.good_mean_diff_min or mean_diff > thresholds.good_mean_diff_max:
        warnings.append(f"intensidade media alterada no limite ({mean_diff:.2f})")

    if max_diff > thresholds.bad_max_diff_max:
        hard.append(f"diferenca maxima excessiva ({max_diff})")
    elif max_diff > thresholds.good_max_diff_max:
        warnings.append(f"diferenca maxima alta ({max_diff})")

    if detection_score < thresholds.bad_detection_score_min:
        hard.append(f"score RetinaFace baixo ({detection_score:.4f})")
    elif detection_score < thresholds.good_detection_score_min:
        warnings.append(f"score RetinaFace apenas razoavel ({detection_score:.4f})")

    if face_ratio < thresholds.bad_face_area_min:
        hard.append(f"rosto demasiado pequeno ({percent(face_ratio)}% do crop/base)")
    elif face_ratio < thresholds.good_face_area_min:
        warnings.append(f"rosto pequeno ({percent(face_ratio)}% do crop/base)")

    if hard:
        grade = "mau"
    elif warnings:
        grade = "mediano"
    else:
        grade = "bom"

    return {
        "grade": grade,
        "hard_issues": hard,
        "warnings": warnings,
        "metrics": {
            "mask_area_percent_aligned": percent(mask_area_ratio),
            "changed_percent_inside_mask": percent(changed_ratio),
            "mean_diff_changed_inside": round(mean_diff, 4),
            "max_diff_inside": max_diff,
            "outside_changed_pixels_total": outside_changed,
            "contour_count": contour_count,
            "retinaface_score": round(detection_score, 5),
            "face_area_percent_reference": percent(face_ratio),
        },
    }


def make_tile(cv2, np, image_path: Path, label: str, size: int = 256):
    image = cv2.imread(str(image_path))
    if image is None:
        image = np.zeros((size, size, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size + 34, size, 3), 245, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, label[:36], (8, size + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def build_per_image_panel(result_dir: Path, summary: dict[str, Any]) -> str | None:
    import cv2
    import numpy as np

    source_paths = [
        (result_dir / "original_image.png", "original"),
        (result_dir / "selected_mask_overlay.png", "mascara"),
        (result_dir / "resultado_final.png", "resultado"),
        (result_dir / "difference_inside_mask.png", "diff dentro"),
    ]
    if not any(path.exists() for path, _ in source_paths):
        return None
    tiles = [make_tile(cv2, np, path, label) for path, label in source_paths]
    panel = cv2.hconcat(tiles)
    grade = summary["grade"].upper()
    color = {"BOM": (40, 140, 40), "MEDIANO": (0, 140, 220), "MAU": (20, 20, 220)}.get(grade, (40, 40, 40))
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 28), color, -1)
    cv2.putText(
        panel,
        f"{grade} | {summary['image_name']}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    panel_path = result_dir / "audit_panel.jpg"
    cv2.imwrite(str(panel_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(panel_path)


def build_contact_sheets(output_dir: Path, rows: list[dict[str, Any]], columns: int) -> list[str]:
    import cv2
    import numpy as np

    contact_dir = output_dir / "contact_sheets"
    if contact_dir.exists():
        shutil.rmtree(contact_dir)

    panels = []
    for row in rows:
        panel_path = row.get("panel_path")
        if panel_path and Path(panel_path).exists():
            image = cv2.imread(str(panel_path))
            if image is not None:
                panels.append((row, image))

    if not panels:
        return []

    panel_h, panel_w = panels[0][1].shape[:2]
    columns = max(1, int(columns))
    pages: list[str] = []
    for grade in ("mau", "mediano", "bom", "all"):
        grade_panels = panels if grade == "all" else [(row, image) for row, image in panels if row["grade"] == grade]
        if not grade_panels:
            continue
        rows_per_page = 4
        per_page = columns * rows_per_page
        for page_index in range(0, len(grade_panels), per_page):
            chunk = grade_panels[page_index : page_index + per_page]
            sheet_rows = []
            for row_start in range(0, len(chunk), columns):
                line = [image for _, image in chunk[row_start : row_start + columns]]
                while len(line) < columns:
                    line.append(np.full((panel_h, panel_w, 3), 245, dtype=np.uint8))
                sheet_rows.append(cv2.hconcat(line))
            sheet = cv2.vconcat(sheet_rows)
            path = contact_dir / f"{grade}_{(page_index // per_page) + 1:02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            pages.append(str(path))
    return pages


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "image_name",
        "image_path",
        "grade",
        "auto_grade",
        "visual_review_note",
        "hard_issues",
        "warnings",
        "mask_area_percent_aligned",
        "changed_percent_inside_mask",
        "mean_diff_changed_inside",
        "max_diff_inside",
        "outside_changed_pixels_total",
        "contour_count",
        "retinaface_score",
        "face_area_percent_reference",
        "output_dir",
        "panel_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or {}
            flat = {
                "index": row["index"],
                "image_name": row["image_name"],
                "image_path": row["image_path"],
                "grade": row["grade"],
                "auto_grade": row.get("auto_grade", row["grade"]),
                "visual_review_note": row.get("visual_review_note", ""),
                "hard_issues": " | ".join(row.get("hard_issues") or []),
                "warnings": " | ".join(row.get("warnings") or []),
                "output_dir": row.get("output_dir", ""),
                "panel_path": row.get("panel_path", ""),
            }
            flat.update(metrics)
            writer.writerow(flat)


def write_markdown_report(output_dir: Path, rows: list[dict[str, Any]], contact_sheets: list[str], description: str) -> None:
    counts = {grade: sum(1 for row in rows if row["grade"] == grade) for grade in ("bom", "mediano", "mau")}
    visual_downgrades = sum(1 for row in rows if row.get("visual_review_note"))
    lines = [
        "# Auditoria rigorosa do dataset",
        "",
        f"Prompt testado: `{description}`",
        "",
        "## Totais",
        "",
        f"- Total: {len(rows)}",
        f"- Bons: {counts['bom']}",
        f"- Medianos: {counts['mediano']}",
        f"- Maus: {counts['mau']}",
        f"- Downgrades por revisao visual: {visual_downgrades}",
        "",
        "## Criterios",
        "",
        "- `mau`: falha de pipeline, alteracao fora da mascara, mascara muito pequena/grande, edicao quase inexistente, FaceMesh/RetinaFace fraco, sem alinhamento ou diferenca maxima extrema.",
        "- `mediano`: tecnicamente passa, mas tem aviso de qualidade: score razoavel, mascara no limite, alteracao moderada ou diferenca maxima alta.",
        "- `bom`: sem problemas duros nem avisos pelos limites definidos no script.",
        "",
        "## Contact sheets",
        "",
    ]
    for sheet in contact_sheets:
        rel = Path(sheet).resolve().relative_to(output_dir.resolve())
        lines.append(f"- [{rel.as_posix()}]({rel.as_posix()})")
    lines.extend(["", "## Maus e medianos", ""])
    for row in [item for item in rows if item["grade"] != "bom"]:
        reasons = row.get("hard_issues") or row.get("warnings") or ["sem motivo registado"]
        lines.append(f"- `{row['grade']}` `{row['image_name']}`: {'; '.join(reasons)}")
    (output_dir / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_image(
    index: int,
    image_path: Path,
    result_dir: Path,
    description: str,
    edit_region: str,
    thresholds: AuditThresholds,
    resume: bool,
) -> dict[str, Any]:
    audit_path = result_dir / "audit_metrics.json"
    if resume and audit_path.exists():
        return load_json(audit_path)

    started = time.time()
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        metadata = run_hybrid_pipeline(
            HybridPipelineConfig(
                input_image=image_path,
                output_dir=result_dir,
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
        validation = metadata["validation_report"]
        primary_face_path = result_dir / "primary_face.json"
        primary_face = load_json(primary_face_path) if primary_face_path.exists() else {}
        classification = classify_metrics(metadata, validation, primary_face, thresholds)
        summary = {
            "index": index,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "output_dir": str(result_dir),
            "duration_seconds": round(time.time() - started, 3),
            **classification,
        }
        panel_path = build_per_image_panel(result_dir, summary)
        if panel_path:
            summary["panel_path"] = panel_path
    except Exception as exc:  # noqa: BLE001 - this is an audit runner; failures are results.
        summary = {
            "index": index,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "output_dir": str(result_dir),
            "duration_seconds": round(time.time() - started, 3),
            "grade": "mau",
            "hard_issues": [f"pipeline falhou: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "metrics": {},
        }
    save_json(audit_path, summary)
    return summary


def apply_review_overrides(rows: list[dict[str, Any]], overrides_path: str) -> list[dict[str, Any]]:
    if not overrides_path:
        return rows

    path = Path(overrides_path).resolve()
    overrides = load_json(path)
    severity = {"bom": 0, "mediano": 1, "mau": 2}
    for row in rows:
        override = overrides.get(row["image_name"])
        if not override:
            continue
        new_grade = str(override.get("grade", "")).strip().lower()
        if new_grade not in severity:
            continue
        old_grade = row["grade"]
        if severity[new_grade] < severity.get(old_grade, 0) and not override.get("allow_upgrade", False):
            continue
        note = str(override.get("note", "revisao visual")).strip()
        row["auto_grade"] = old_grade
        row["grade"] = new_grade
        row["visual_review_note"] = note
        reason = f"revisao visual: {note}"
        if new_grade == "mau":
            hard = list(row.get("hard_issues") or [])
            if reason not in hard:
                hard.append(reason)
            row["hard_issues"] = hard
        elif new_grade == "mediano":
            warnings = list(row.get("warnings") or [])
            if reason not in warnings:
                warnings.append(reason)
            row["warnings"] = warnings
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        image_paths = list_dataset_images(Path(args.input_dir))
    else:
        image_paths = list_dataset_images()
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    thresholds = AuditThresholds()
    rows: list[dict[str, Any]] = []
    total = len(image_paths)
    print(f"Auditoria: {total} imagens | prompt: {args.description}")
    for index, image_path in enumerate(image_paths, start=1):
        result_dir = output_dir / "results" / slug_for_image(index, image_path)
        print(f"[{index:03d}/{total:03d}] {image_path.name}", flush=True)
        rows.append(
            audit_image(
                index=index,
                image_path=image_path,
                result_dir=result_dir,
                description=args.description,
                edit_region=args.edit_region,
                thresholds=thresholds,
                resume=args.resume,
            )
        )
        grade = rows[-1]["grade"].upper()
        issues = rows[-1].get("hard_issues") or rows[-1].get("warnings") or []
        suffix = f" - {issues[0]}" if issues else ""
        print(f"    -> {grade}{suffix}", flush=True)

    rows = apply_review_overrides(rows, args.review_overrides)
    grade_order = {"mau": 0, "mediano": 1, "bom": 2}
    rows.sort(key=lambda row: (grade_order.get(row["grade"], 9), row["index"]))
    save_json(output_dir / "audit_summary.json", rows)
    write_csv(output_dir / "audit_summary.csv", rows)
    contact_sheets = build_contact_sheets(output_dir, rows, args.contact_sheet_width)
    write_markdown_report(output_dir, rows, contact_sheets, args.description)

    counts = {grade: sum(1 for row in rows if row["grade"] == grade) for grade in ("bom", "mediano", "mau")}
    print("")
    print("Resumo:")
    print(f"  Bons: {counts['bom']}")
    print(f"  Medianos: {counts['mediano']}")
    print(f"  Maus: {counts['mau']}")
    print(f"  Relatorio: {output_dir / 'AUDIT_REPORT.md'}")
    print(f"  CSV: {output_dir / 'audit_summary.csv'}")


if __name__ == "__main__":
    main()
