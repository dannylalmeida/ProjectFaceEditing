from __future__ import annotations

import subprocess
import os
from pathlib import Path

from src.segmentation.mask_utils import build_soft_mask


def _opencv_edge_inpainting(final_aligned_bgr, mask_uint8, cv2, np, steps: int, strength: float):
    if steps <= 0 or strength <= 0 or not np.any(mask_uint8):
        return final_aligned_bgr, np.zeros_like(mask_uint8), {"backend_used": "none"}

    radius = max(2, min(9, int(round(steps / 5))))
    hard_mask = (mask_uint8 > 16).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    inner = cv2.erode(hard_mask, kernel, iterations=1)
    outer = cv2.dilate(hard_mask, kernel, iterations=1)
    edge_ring = cv2.subtract(outer, inner)
    if not np.any(edge_ring):
        return final_aligned_bgr, edge_ring, {"backend_used": "none"}

    inpainted = cv2.inpaint(final_aligned_bgr, edge_ring, radius, cv2.INPAINT_TELEA)
    alpha = build_soft_mask(edge_ring, radius, cv2, np) * float(max(0.0, min(strength, 1.0)))
    refined = (
        inpainted.astype(np.float32) * alpha
        + final_aligned_bgr.astype(np.float32) * (1.0 - alpha)
    )
    return refined.clip(0, 255).astype(np.uint8), edge_ring, {
        "backend_used": "opencv",
        "radius": radius,
        "edge_pixels": int((edge_ring > 0).sum()),
    }


def _save_rgb(path: Path, image_rgb) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def _to_yaml_path(path: Path, root: Path | None = None) -> str:
    resolved = path.resolve()
    if root is not None:
        return os.path.relpath(str(resolved), str(root.resolve())).replace("\\", "/")
    return str(resolved).replace("\\", "/")


def _resolve_device(device: str, conda_env: str) -> str:
    if device != "auto":
        return device
    completed = subprocess.run(
        ["conda", "run", "-n", conda_env, "python", "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
        capture_output=True,
        text=True,
        check=False,
    )
    resolved = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return resolved if resolved in {"cpu", "cuda"} else "cpu"


def _real_repaint_refinement(
    final_aligned_bgr,
    mask_uint8,
    cv2,
    np,
    steps: int,
    strength: float,
    work_dir: Path,
    project_dir: Path,
    conda_env: str,
    device: str,
):
    if steps <= 0 or strength <= 0 or not np.any(mask_uint8):
        return final_aligned_bgr, np.zeros_like(mask_uint8), {"backend_used": "none"}

    repaint_root = project_dir / "third_party" / "RePaint"
    repaint_model = repaint_root / "data" / "pretrained" / "celeba256_250000.pt"
    if not repaint_root.exists():
        raise FileNotFoundError(f"RePaint repo not found: {repaint_root}")
    if not repaint_model.exists():
        raise FileNotFoundError(f"RePaint checkpoint not found: {repaint_model}")

    radius = max(2, min(9, int(round(steps / 5))))
    hard_mask = (mask_uint8 > 16).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    inner = cv2.erode(hard_mask, kernel, iterations=1)
    outer = cv2.dilate(hard_mask, kernel, iterations=1)
    edge_ring = cv2.subtract(outer, inner)
    if not np.any(edge_ring):
        return final_aligned_bgr, edge_ring, {"backend_used": "none"}

    stage_dir = work_dir / "repaint_real"
    input_dir = stage_dir / "01_inputs"
    gt_dir = input_dir / "gts"
    keep_dir = input_dir / "gt_keep_masks"
    result_dir = stage_dir / "02_repaint"
    inpainted_dir = result_dir / "inpainted"
    masked_dir = result_dir / "gt_masked"
    gt_out_dir = result_dir / "gt"
    keep_out_dir = result_dir / "gt_keep_mask"
    config_path = stage_dir / "repaint_refine.yml"

    for path in (gt_dir, keep_dir, inpainted_dir, masked_dir, gt_out_dir, keep_out_dir):
        path.mkdir(parents=True, exist_ok=True)

    face_256_bgr = cv2.resize(final_aligned_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    edge_256 = cv2.resize(edge_ring, (256, 256), interpolation=cv2.INTER_NEAREST)
    keep_mask_256 = np.where(edge_256 > 0, 0, 255).astype(np.uint8)
    _save_rgb(gt_dir / "face.png", cv2.cvtColor(face_256_bgr, cv2.COLOR_BGR2RGB))
    _save_rgb(keep_dir / "face.png", np.repeat(keep_mask_256[..., None], 3, axis=2))

    resolved_device = _resolve_device(device, conda_env)
    jump_length = 10
    jump_samples = 2 if steps <= 25 else 4
    repaint_steps = max(2, int(steps))
    yaml = f"""attention_resolutions: 32,16,8
class_cond: false
diffusion_steps: 1000
learn_sigma: true
noise_schedule: linear
num_channels: 256
num_head_channels: 64
num_heads: 4
num_res_blocks: 2
resblock_updown: true
use_fp16: false
use_scale_shift_norm: true
classifier_scale: 4.0
lr_kernel_n_std: 2
num_samples: 1
show_progress: false
timestep_respacing: '{repaint_steps}'
use_kl: false
predict_xstart: false
rescale_timesteps: false
rescale_learned_sigmas: false
classifier_use_fp16: false
classifier_width: 128
classifier_depth: 2
classifier_attention_resolutions: 32,16,8
classifier_use_scale_shift_norm: true
classifier_resblock_updown: true
classifier_pool: attention
num_heads_upsample: -1
channel_mult: ''
dropout: 0.0
use_checkpoint: false
use_new_attention_order: false
clip_denoised: true
use_ddim: false
latex_name: RePaint
method_name: Repaint
image_size: 256
model_path: '{_to_yaml_path(repaint_model, repaint_root)}'
name: hybrid_repaint_refine
device: {resolved_device}
inpa_inj_sched_prev: true
n_jobs: 0
print_estimated_vars: true
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: {repaint_steps}
  n_sample: 1
  jump_length: {jump_length}
  jump_n_sample: {jump_samples}
data:
  eval:
    project_face_mask:
      mask_loader: true
      gt_path: '{_to_yaml_path(gt_dir, repaint_root)}'
      mask_path: '{_to_yaml_path(keep_dir, repaint_root)}'
      image_size: 256
      class_cond: false
      deterministic: true
      random_crop: false
      random_flip: false
      return_dict: true
      drop_last: false
      batch_size: 1
      return_dataloader: true
      offset: 0
      max_len: 1
      paths:
        srs: '{_to_yaml_path(inpainted_dir, repaint_root)}'
        lrs: '{_to_yaml_path(masked_dir, repaint_root)}'
        gts: '{_to_yaml_path(gt_out_dir, repaint_root)}'
        gt_keep_masks: '{_to_yaml_path(keep_out_dir, repaint_root)}'
"""
    config_path.write_text(yaml, encoding="utf-8")

    completed = subprocess.run(
        ["conda", "run", "-n", conda_env, "python", "test.py", "--conf_path", str(config_path)],
        cwd=str(repaint_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "RePaint failed").strip())

    result_path = inpainted_dir / "face.png"
    if not result_path.exists():
        raise FileNotFoundError(f"RePaint result not found: {result_path}")

    repainted_256_bgr = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
    if repainted_256_bgr is None:
        raise FileNotFoundError(f"Could not read RePaint result: {result_path}")
    repainted_bgr = cv2.resize(
        repainted_256_bgr,
        (final_aligned_bgr.shape[1], final_aligned_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    alpha = build_soft_mask(edge_ring, radius, cv2, np) * float(max(0.0, min(strength, 1.0)))
    refined = (
        repainted_bgr.astype(np.float32) * alpha
        + final_aligned_bgr.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)
    return refined, edge_ring, {
        "backend_used": "repaint",
        "config_path": str(config_path),
        "result_path": str(result_path),
        "edge_pixels": int((edge_ring > 0).sum()),
        "device": resolved_device,
    }


def run_repaint_or_inpainting(
    final_aligned_bgr,
    mask_uint8,
    cv2,
    np,
    steps: int = 20,
    strength: float = 0.35,
    backend: str = "opencv",
    work_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
    conda_env: str = "styleclip",
    device: str = "auto",
):
    if backend == "repaint":
        try:
            if work_dir is None or project_dir is None:
                raise ValueError("work_dir and project_dir are required for backend='repaint'")
            return _real_repaint_refinement(
                final_aligned_bgr,
                mask_uint8,
                cv2,
                np,
                steps=steps,
                strength=strength,
                work_dir=Path(work_dir),
                project_dir=Path(project_dir),
                conda_env=conda_env,
                device=device,
            )
        except Exception as exc:
            refined, edge_ring, info = _opencv_edge_inpainting(final_aligned_bgr, mask_uint8, cv2, np, steps, strength)
            info["backend_requested"] = "repaint"
            info["fallback_reason"] = str(exc)
            return refined, edge_ring, info

    refined, edge_ring, info = _opencv_edge_inpainting(final_aligned_bgr, mask_uint8, cv2, np, steps, strength)
    info["backend_requested"] = backend
    return refined, edge_ring, info
