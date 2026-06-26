"""Seleciona imagens boas para edicao local de boca, nariz e olhos.

Usa o mediapipe FaceMesh (a mesma deteccao que o pipeline de edicao usa para
ancoras de boca/nariz e iris). Uma imagem so e considerada "pronta para edicao"
se a face for detetada com confianca, estiver frontal, tiver tamanho adequado e
as tres regioes (boca, nariz, olhos+iris) estiverem visiveis e nao cortadas.

Saida: uma pasta com hardlinks para as imagens escolhidas (custo de disco ~0 na
mesma drive) + um manifesto JSON/CSV com as metricas de cada imagem.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


DEFAULT_INPUT_DIR = PROJECT_DIR / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "dataset_edit_ready"

# Indices de iris quando refine_landmarks=True (5 por olho).
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
# Cantos/parpebras para abertura do olho (eye aspect ratio).
RIGHT_EYE = {"top": 159, "bottom": 145, "left": 33, "right": 133}
LEFT_EYE = {"top": 386, "bottom": 374, "left": 362, "right": 263}
NOSE_TIP = 1
FACE_RIGHT = 234   # bordo direito do rosto (lado esquerdo da imagem)
FACE_LEFT = 454    # bordo esquerdo do rosto (lado direito da imagem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curadoria de imagens prontas para edicao de boca/nariz/olhos.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", type=int, default=1000, help="Numero ideal de imagens a selecionar.")
    parser.add_argument("--min", type=int, default=800, help="Minimo aceitavel; abaixo disto avisa.")
    parser.add_argument("--limit", type=int, default=0, help="Processa so as primeiras N imagens (smoke test).")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "manifest"),
        default="hardlink",
        help="hardlink (disco ~0), copy (duplica), manifest (so a lista).",
    )
    parser.add_argument("--min-face-area", type=float, default=0.10, help="Area minima da face (fracao da imagem).")
    parser.add_argument("--max-yaw", type=float, default=0.22, help="Desvio horizontal maximo (0=perfeitamente frontal).")
    parser.add_argument("--max-roll-deg", type=float, default=14.0, help="Inclinacao maxima da linha dos olhos.")
    parser.add_argument("--min-eye-open", type=float, default=0.15, help="Abertura minima dos olhos (EAR).")
    parser.add_argument("--margin", type=float, default=0.02, help="Margem de seguranca nas bordas (fracao).")
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)


def _xy(landmarks, index, w, h):
    lm = landmarks[index]
    return (float(lm.x) * w, float(lm.y) * h)


def _region_bbox(landmarks, indices, w, h):
    xs = [float(landmarks[i].x) * w for i in indices if i < len(landmarks)]
    ys = [float(landmarks[i].y) * h for i in indices if i < len(landmarks)]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _eye_aspect_ratio(landmarks, eye, w, h):
    top = _xy(landmarks, eye["top"], w, h)
    bottom = _xy(landmarks, eye["bottom"], w, h)
    left = _xy(landmarks, eye["left"], w, h)
    right = _xy(landmarks, eye["right"], w, h)
    vertical = abs(top[1] - bottom[1])
    horizontal = abs(left[0] - right[0])
    if horizontal < 1e-3:
        return 0.0
    return vertical / horizontal


def evaluate(image_bgr, face_mesh, mp, cv2, np, args) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return {"ok": False, "reason": "sem face detetada"}

    landmarks = results.multi_face_landmarks[0].landmark
    if len(landmarks) < 478:
        return {"ok": False, "reason": "iris nao detetada (refine incompleto)"}

    nose_idx = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_NOSE for i in edge})
    lips_idx = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_LIPS for i in edge})
    leye_idx = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_LEFT_EYE for i in edge})
    reye_idx = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE for i in edge})

    # Area da face a partir do bbox de todos os landmarks.
    all_x = [float(p.x) for p in landmarks]
    all_y = [float(p.y) for p in landmarks]
    face_w = max(all_x) - min(all_x)
    face_h = max(all_y) - min(all_y)
    face_area = max(0.0, face_w) * max(0.0, face_h)
    if face_area < args.min_face_area:
        return {"ok": False, "reason": f"rosto pequeno ({face_area:.3f})", "face_area": face_area}

    # Todas as regioes-alvo dentro da imagem com margem (nao cortadas).
    margin = args.margin
    lo, hi = margin, 1.0 - margin
    for name, idx in (("nariz", nose_idx), ("boca", lips_idx), ("olho_esq", leye_idx), ("olho_dir", reye_idx)):
        box = _region_bbox(landmarks, idx, 1.0, 1.0)
        if box is None:
            return {"ok": False, "reason": f"{name} sem landmarks"}
        x1, y1, x2, y2 = box
        if x1 < lo or y1 < lo or x2 > hi or y2 > hi:
            return {"ok": False, "reason": f"{name} cortada na borda"}

    # Frontalidade: posicao do nariz entre os bordos do rosto (ideal 0.5).
    nose_x, _ = _xy(landmarks, NOSE_TIP, 1.0, 1.0)
    right_x, _ = _xy(landmarks, FACE_RIGHT, 1.0, 1.0)
    left_x, _ = _xy(landmarks, FACE_LEFT, 1.0, 1.0)
    span = abs(left_x - right_x)
    if span < 1e-3:
        return {"ok": False, "reason": "perfil extremo"}
    yaw = abs(((nose_x - right_x) / span) - 0.5) * 2.0  # 0=frontal, 1=perfil
    if yaw > args.max_yaw:
        return {"ok": False, "reason": f"nao frontal (yaw={yaw:.2f})", "yaw": yaw}

    # Roll: angulo da linha entre os centros dos olhos.
    rc = _xy(landmarks, RIGHT_EYE["left"], 1.0, 1.0)
    lc = _xy(landmarks, LEFT_EYE["right"], 1.0, 1.0)
    roll_deg = abs(np.degrees(np.arctan2(lc[1] - rc[1], (lc[0] - rc[0]) + 1e-9)))
    roll_deg = min(roll_deg, 180.0 - roll_deg)
    if roll_deg > args.max_roll_deg:
        return {"ok": False, "reason": f"inclinada (roll={roll_deg:.1f})", "roll": roll_deg}

    # Olhos abertos (importante para edicao de olhos/iris).
    ear_r = _eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ear_l = _eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
    eye_open = min(ear_r, ear_l)
    if eye_open < args.min_eye_open:
        return {"ok": False, "reason": f"olhos fechados (ear={eye_open:.2f})", "eye_open": eye_open}

    # Pontuacao de qualidade: frontal + face grande + olhos bem abertos + nivelada.
    score = (
        (1.0 - min(yaw / max(args.max_yaw, 1e-6), 1.0)) * 0.45
        + min(face_area / 0.45, 1.0) * 0.25
        + min(eye_open / 0.30, 1.0) * 0.20
        + (1.0 - min(roll_deg / max(args.max_roll_deg, 1e-6), 1.0)) * 0.10
    )
    return {
        "ok": True,
        "score": round(float(score), 4),
        "face_area": round(float(face_area), 4),
        "yaw": round(float(yaw), 4),
        "roll_deg": round(float(roll_deg), 2),
        "eye_open": round(float(eye_open), 4),
    }


def materialize(selected: list[dict], output_dir: Path, link_mode: str) -> None:
    if link_mode == "manifest":
        return
    if output_dir.exists():
        # Limpa apenas os ficheiros de imagem antigos (nao mexe noutra coisa).
        for old in output_dir.glob("*.png"):
            old.unlink()
        for old in output_dir.glob("*.jpg"):
            old.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        src = Path(row["image_path"])
        dst = output_dir / src.name
        if dst.exists():
            dst.unlink()
        if link_mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError:
                # Drives diferentes ou FS sem hardlink -> copia.
                import shutil
                shutil.copyfile(src, dst)
        else:
            import shutil
            shutil.copyfile(src, dst)


def main() -> None:
    args = parse_args()
    import cv2
    import numpy as np
    import mediapipe as mp

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    images = list_images(input_dir)
    if args.limit > 0:
        images = images[: args.limit]
    total = len(images)
    print(f"A avaliar {total} imagens de {input_dir}", flush=True)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    passed: list[dict] = []
    rejected = 0
    reasons: dict[str, int] = {}
    started = time.time()
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            rejected += 1
            reasons["ilegivel"] = reasons.get("ilegivel", 0) + 1
            continue
        verdict = evaluate(image, face_mesh, mp, cv2, np, args)
        if verdict.get("ok"):
            passed.append({"image_name": image_path.name, "image_path": str(image_path), **verdict})
        else:
            rejected += 1
            key = str(verdict.get("reason", "?")).split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        if index % 250 == 0 or index == total:
            rate = index / max(1e-6, time.time() - started)
            print(f"  [{index}/{total}] aprovadas={len(passed)} rejeitadas={rejected} ({rate:.1f} img/s)", flush=True)

    face_mesh.close()

    passed.sort(key=lambda r: r["score"], reverse=True)
    selected = passed[: args.target] if args.target > 0 else passed

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir if args.link_mode != "manifest" else output_dir.parent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = manifest_dir / "edit_ready_manifest.json"
    manifest_csv = manifest_dir / "edit_ready_manifest.csv"

    materialize(selected, output_dir, args.link_mode)

    manifest_json.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "link_mode": args.link_mode,
                "total_scanned": total,
                "total_passed": len(passed),
                "total_selected": len(selected),
                "thresholds": {
                    "min_face_area": args.min_face_area,
                    "max_yaw": args.max_yaw,
                    "max_roll_deg": args.max_roll_deg,
                    "min_eye_open": args.min_eye_open,
                    "margin": args.margin,
                },
                "rejection_reasons": reasons,
                "images": [r["image_name"] for r in selected],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "image_path", "score", "face_area", "yaw", "roll_deg", "eye_open"])
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    print("")
    print("=== RESUMO ===")
    print(f"  Avaliadas: {total}")
    print(f"  Aprovadas (passaram nos criterios): {len(passed)}")
    print(f"  Selecionadas (ate ao alvo {args.target}): {len(selected)}")
    print(f"  Modo: {args.link_mode} -> {output_dir}")
    print(f"  Manifesto: {manifest_json}")
    if reasons:
        print("  Motivos de rejeicao:")
        for reason, count in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    - {reason}: {count}")
    if len(selected) < args.min:
        print("")
        print(f"  AVISO: so consegui {len(selected)} imagens boas, abaixo do minimo de {args.min}.")
        print("  Se tens uma pasta com mais imagens, indica-a com --input-dir para eu varrer tambem.")
    elif len(selected) < args.target:
        print("")
        print(f"  Nota: {len(selected)} imagens boas (>= minimo {args.min}, mas abaixo do ideal {args.target}).")
        print("  Para chegar ao ideal, podes baixar os limites ou indicar uma pasta extra com --input-dir.")


if __name__ == "__main__":
    main()
