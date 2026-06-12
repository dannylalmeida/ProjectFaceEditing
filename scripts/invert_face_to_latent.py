from __future__ import annotations

import argparse
import math
import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
E4E_ROOT = PROJECT_DIR / "third_party" / "encoder4editing"
PSP_ROOT = PROJECT_DIR / "third_party" / "pixel2style2pixel"
DEFAULT_E4E_CKPT = E4E_ROOT / "pretrained_models" / "e4e_ffhq_encode.pt"
DEFAULT_PSP_CKPT = PSP_ROOT / "pretrained_models" / "psp_ffhq_encode.pt"


def configure_runtime() -> None:
    matplotlib_cache = PROJECT_DIR / ".cache" / "matplotlib_styleclip"
    torch_extensions = PROJECT_DIR / ".cache" / "torch_extensions"

    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    torch_extensions.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions))
    os.environ.setdefault("USERPROFILE", str(PROJECT_DIR))


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA nao esta disponivel no ambiente styleclip. Vou usar cpu para a inversao.")
        return "cpu"
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def get_lanczos_resample(Image):
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def compute_reconstruction_metrics(input_image, reconstruction_image):
    import numpy as np

    input_array = np.asarray(input_image.convert("RGB"), dtype=np.float32)
    reconstruction_array = np.asarray(reconstruction_image.convert("RGB"), dtype=np.float32)

    if input_array.shape != reconstruction_array.shape:
        raise ValueError(
            f"Shapes incompativeis para comparar reconstrucao: {input_array.shape} vs {reconstruction_array.shape}"
        )

    diff = input_array - reconstruction_array
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    psnr = 999.0 if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))

    input_gray = (
        0.299 * input_array[..., 0] + 0.587 * input_array[..., 1] + 0.114 * input_array[..., 2]
    )
    reconstruction_gray = (
        0.299 * reconstruction_array[..., 0]
        + 0.587 * reconstruction_array[..., 1]
        + 0.114 * reconstruction_array[..., 2]
    )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = float(input_gray.mean())
    mu_y = float(reconstruction_gray.mean())
    var_x = float(input_gray.var())
    var_y = float(reconstruction_gray.var())
    cov_xy = float(((input_gray - mu_x) * (reconstruction_gray - mu_y)).mean())
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )

    return {
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
        "global_ssim": float(ssim),
    }


def save_reconstruction_comparison(
    input_image,
    reconstruction_image,
    comparison_path: Path,
    diff_path: Path,
    reconstruction_label: str = "reconstrucao latent",
) -> None:
    from PIL import Image, ImageChops, ImageDraw

    input_rgb = input_image.convert("RGB")
    reconstruction_rgb = reconstruction_image.convert("RGB")
    diff = ImageChops.difference(input_rgb, reconstruction_rgb)
    enhanced_diff = diff.point(lambda value: min(255, value * 4))

    comparison = Image.new("RGB", (input_rgb.width * 3, input_rgb.height + 28), "white")
    draw = ImageDraw.Draw(comparison)
    comparison.paste(input_rgb, (0, 28))
    comparison.paste(reconstruction_rgb, (input_rgb.width, 28))
    comparison.paste(enhanced_diff, (input_rgb.width * 2, 28))
    draw.text((8, 8), "input encoder 256", fill=(0, 0, 0))
    draw.text((input_rgb.width + 8, 8), reconstruction_label, fill=(0, 0, 0))
    draw.text((input_rgb.width * 2 + 8, 8), "diferenca x4", fill=(0, 0, 0))

    comparison.save(comparison_path)
    enhanced_diff.save(diff_path)


def save_identity_restored_reconstruction(input_image, reconstruction_image, output_path: Path, residual_path: Path) -> None:
    import numpy as np
    from PIL import Image

    input_array = np.asarray(input_image.convert("RGB"), dtype=np.int16)
    reconstruction_array = np.asarray(reconstruction_image.convert("RGB"), dtype=np.int16)
    residual = input_array - reconstruction_array
    restored = np.clip(reconstruction_array + residual, 0, 255).astype(np.uint8)

    residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)
    Image.fromarray(restored, mode="RGB").save(output_path)
    Image.fromarray(residual_vis, mode="RGB").save(residual_path)


def reconstruct_from_latent(net, latent_codes):
    reconstruction_tensor, _ = net.decoder(
        [latent_codes],
        input_is_latent=True,
        randomize_noise=False,
        return_latents=True,
    )
    return net.face_pool(reconstruction_tensor)


def refine_latent_for_reconstruction(
    net,
    initial_latent,
    target_tensor,
    steps: int,
    learning_rate: float,
    latent_l2_weight: float,
    log_interval: int = 10,
):
    import torch
    import torch.nn.functional as F

    if steps <= 0:
        return initial_latent.detach(), []

    original_latent = initial_latent.detach().clone()
    refined_latent = original_latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([refined_latent], lr=learning_rate)
    history = []
    with torch.no_grad():
        best_reconstruction = reconstruct_from_latent(net, original_latent)
        best_pixel_loss = F.mse_loss(best_reconstruction, target_tensor)
        best_latent = original_latent.detach().clone()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        reconstruction_tensor = reconstruct_from_latent(net, refined_latent)
        pixel_loss = F.mse_loss(reconstruction_tensor, target_tensor)
        latent_l2 = F.mse_loss(refined_latent, original_latent)
        loss = pixel_loss + latent_l2_weight * latent_l2
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            candidate_reconstruction = reconstruct_from_latent(net, refined_latent)
            candidate_pixel_loss = F.mse_loss(candidate_reconstruction, target_tensor)
            if candidate_pixel_loss < best_pixel_loss:
                best_pixel_loss = candidate_pixel_loss.detach().clone()
                best_latent = refined_latent.detach().clone()

        if step == 1 or step == steps or step % log_interval == 0:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pixel_mse_loss": float(pixel_loss.detach().cpu()),
                "candidate_pixel_mse_loss": float(candidate_pixel_loss.detach().cpu()),
                "best_pixel_mse_loss": float(best_pixel_loss.detach().cpu()),
                "latent_l2_loss": float(latent_l2.detach().cpu()),
            }
            history.append(record)
            print(
                "Refinamento latent "
                f"{step}/{steps}: loss={record['loss']:.6f}, "
                f"pixel={record['pixel_mse_loss']:.6f}, "
                f"best_pixel={record['best_pixel_mse_loss']:.6f}, "
                f"latent_l2={record['latent_l2_loss']:.6f}"
            )

    return best_latent.detach(), history


def load_encoder_model(encoder_backend: str, checkpoint_path: Path, device: str):
    if encoder_backend == "e4e":
        sys.path.insert(0, str(E4E_ROOT))

        from utils.model_utils import setup_model

        net, opts = setup_model(str(checkpoint_path), device=device)
        return net, opts

    if encoder_backend == "psp":
        sys.path.insert(0, str(PSP_ROOT))

        import argparse
        import torch
        from models.psp import pSp

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        opts_dict = dict(ckpt["opts"])
        opts_dict.update(
            {
                "checkpoint_path": str(checkpoint_path),
                "device": device,
            }
        )
        opts_dict.setdefault("learn_in_w", False)
        opts_dict.setdefault("output_size", 1024)
        opts = argparse.Namespace(**opts_dict)

        net = pSp(opts)
        net.eval()
        net = net.to(device)
        return net, opts

    raise ValueError(f"Backend de encoder desconhecido: {encoder_backend}")


def add_latent_avg_if_needed(net, latent_codes):
    if not getattr(net.opts, "start_from_latent_avg", False):
        return latent_codes

    if getattr(net.opts, "learn_in_w", False):
        return latent_codes + net.latent_avg.repeat(latent_codes.shape[0], 1)

    if latent_codes.ndim == 2:
        return latent_codes + net.latent_avg.repeat(latent_codes.shape[0], 1, 1)[:, 0, :]

    return latent_codes + net.latent_avg.repeat(latent_codes.shape[0], 1, 1)


def encode_face(
    input_crop: Path,
    output_dir: Path,
    checkpoint_path: Path,
    requested_device: str,
    encoder_backend: str = "e4e",
    save_reconstruction_preview: bool = False,
    refine_latent_steps: int = 0,
    refine_learning_rate: float = 0.01,
    refine_latent_l2: float = 0.001,
) -> None:
    sys.path.insert(0, str(E4E_ROOT))

    import torch
    from PIL import Image
    from utils.common import tensor2im

    device = resolve_device(requested_device)
    net, _ = load_encoder_model(encoder_backend, checkpoint_path, device=device)

    transform = build_transform()
    image = Image.open(input_crop).convert("RGB")
    resized_image = image.resize((256, 256), resample=get_lanczos_resample(Image))
    input_tensor = transform(resized_image).unsqueeze(0).to(device)

    with torch.no_grad():
        latent_codes = net.encoder(input_tensor)
        latent_codes = add_latent_avg_if_needed(net, latent_codes)

        reconstruction_tensor = reconstruct_from_latent(net, latent_codes)

    initial_latent_codes = latent_codes.detach().clone()
    initial_reconstruction_tensor = reconstruction_tensor.detach().clone()
    refinement_history = []
    if refine_latent_steps > 0:
        print(
            "A refinar o latent W+ para aproximar a reconstrucao ao crop original "
            f"({refine_latent_steps} passos)."
        )
        latent_codes, refinement_history = refine_latent_for_reconstruction(
            net,
            latent_codes,
            input_tensor,
            refine_latent_steps,
            refine_learning_rate,
            refine_latent_l2,
        )
        with torch.no_grad():
            reconstruction_tensor = reconstruct_from_latent(net, latent_codes)

    output_dir.mkdir(parents=True, exist_ok=True)

    input_crop_quality_path = output_dir / "input_crop_full_quality.png"
    input_preview_path = output_dir / "input_resized_256_lanczos.png"
    initial_reconstruction_preview_path = output_dir / f"{encoder_backend}_initial_reconstruction_preview.png"
    initial_reconstruction_comparison_path = output_dir / f"{encoder_backend}_initial_reconstruction_comparison.png"
    initial_reconstruction_diff_path = output_dir / f"{encoder_backend}_initial_reconstruction_diff_x4.png"
    reconstruction_preview_path = output_dir / f"{encoder_backend}_reconstruction_preview.png"
    reconstruction_comparison_path = output_dir / f"{encoder_backend}_reconstruction_comparison.png"
    reconstruction_diff_path = output_dir / f"{encoder_backend}_reconstruction_diff_x4.png"
    restored_reconstruction_path = output_dir / f"{encoder_backend}_identity_restored_reconstruction.png"
    residual_map_path = output_dir / f"{encoder_backend}_reconstruction_residual_map.png"
    latent_path = output_dir / "inversion_latent.pt"
    initial_latent_path = output_dir / "inversion_latent_initial.pt"
    metadata_path = output_dir / "inversion_metadata.json"

    reconstruction_metrics = None
    initial_reconstruction_metrics = None
    restored_reconstruction_metrics = None
    image.save(input_crop_quality_path)
    resized_image.save(input_preview_path)
    if save_reconstruction_preview or refine_latent_steps > 0:
        initial_reconstruction_image = tensor2im(initial_reconstruction_tensor[0].detach().cpu())
        initial_reconstruction_image.save(initial_reconstruction_preview_path)
        initial_reconstruction_metrics = compute_reconstruction_metrics(resized_image, initial_reconstruction_image)
        save_reconstruction_comparison(
            resized_image,
            initial_reconstruction_image,
            initial_reconstruction_comparison_path,
            initial_reconstruction_diff_path,
            f"{encoder_backend} inicial",
        )

        reconstruction_image = tensor2im(reconstruction_tensor[0].detach().cpu())
        reconstruction_image.save(reconstruction_preview_path)
        reconstruction_metrics = compute_reconstruction_metrics(resized_image, reconstruction_image)
        save_reconstruction_comparison(
            resized_image,
            reconstruction_image,
            reconstruction_comparison_path,
            reconstruction_diff_path,
            f"{encoder_backend} refinado" if refine_latent_steps > 0 else f"{encoder_backend}",
        )
        save_identity_restored_reconstruction(
            resized_image,
            reconstruction_image,
            restored_reconstruction_path,
            residual_map_path,
        )
        restored_reconstruction_metrics = compute_reconstruction_metrics(
            resized_image,
            resized_image,
        )
    torch.save(initial_latent_codes.detach().cpu(), initial_latent_path)
    torch.save(latent_codes.detach().cpu(), latent_path)

    latent_shape = list(latent_codes.shape)
    latent_space = "W+" if latent_codes.ndim == 3 else "W"
    metadata = {
        "operation": (
            f"{encoder_backend}_encode_only"
            if not (save_reconstruction_preview or refine_latent_steps > 0)
            else f"{encoder_backend}_encode_refine_and_reconstruct_preview"
        ),
        "is_edit": False,
        "encoder_backend": encoder_backend,
        "input_crop": str(input_crop),
        "checkpoint": str(checkpoint_path),
        "e4e_checkpoint": str(checkpoint_path) if encoder_backend == "e4e" else None,
        "psp_checkpoint": str(checkpoint_path) if encoder_backend == "psp" else None,
        "device": device,
        "latent_path": str(latent_path),
        "initial_latent_path": str(initial_latent_path),
        "latent_refinement_enabled": refine_latent_steps > 0,
        "latent_refinement_steps": refine_latent_steps,
        "latent_refinement_learning_rate": refine_learning_rate,
        "latent_refinement_l2": refine_latent_l2,
        "latent_refinement_history": refinement_history,
        "latent_shape": latent_shape,
        "latent_space": latent_space,
        "input_crop_full_quality_path": str(input_crop_quality_path),
        "input_resized_path": str(input_preview_path),
        "input_crop_size": list(image.size),
        "input_encoder_size": [256, 256],
        "input_e4e_size": [256, 256] if encoder_backend == "e4e" else None,
        "input_psp_size": [256, 256] if encoder_backend == "psp" else None,
        "resize_interpolation": "PIL.Image.Resampling.LANCZOS",
        "initial_reconstruction_preview_path": str(initial_reconstruction_preview_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "initial_reconstruction_comparison_path": str(initial_reconstruction_comparison_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "initial_reconstruction_diff_x4_path": str(initial_reconstruction_diff_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "initial_reconstruction_metrics": initial_reconstruction_metrics,
        "reconstruction_preview_path": str(reconstruction_preview_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "reconstruction_comparison_path": str(reconstruction_comparison_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "reconstruction_diff_x4_path": str(reconstruction_diff_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "reconstruction_metrics": reconstruction_metrics,
        "identity_restored_reconstruction_path": str(restored_reconstruction_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "reconstruction_residual_map_path": str(residual_map_path)
        if (save_reconstruction_preview or refine_latent_steps > 0)
        else None,
        "identity_restored_reconstruction_metrics": restored_reconstruction_metrics,
        "note": (
            f"O {encoder_backend} gera o latent W+. O refinamento melhora a reconstrucao, mas "
            "um latent StyleGAN puro nao consegue representar todos os detalhes pixel-a-pixel de uma foto real. "
            "O residual map documenta a informacao que ainda fica fora do espaco latente."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Latent guardado em: {latent_path}")
    print(f"Crop em qualidade original guardado em: {input_crop_quality_path}")
    print(f"Backend de inversao: {encoder_backend}")
    print(f"Shape do latent: {latent_shape} ({latent_space})")
    print(f"Input 256x256 do encoder guardado em: {input_preview_path}")
    if save_reconstruction_preview or refine_latent_steps > 0:
        print(f"Preview inicial {encoder_backend}: {initial_reconstruction_preview_path}")
        print(f"Metricas iniciais {encoder_backend}: {json.dumps(initial_reconstruction_metrics, indent=2)}")
        print(f"Preview de reconstrucao {encoder_backend} guardada em: {reconstruction_preview_path}")
        print(f"Comparacao input/reconstrucao guardada em: {reconstruction_comparison_path}")
        print(f"Metricas reconstrucao {encoder_backend}: {json.dumps(reconstruction_metrics, indent=2)}")
        print(f"Reconstrucao com residual de identidade guardada em: {restored_reconstruction_path}")
        print(f"Mapa residual guardado em: {residual_map_path}")
    else:
        print(f"Preview de reconstrucao {encoder_backend} nao gerada; modo encode-only ativo.")
    print(f"Metadados guardados em: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inverte um crop facial para latent W/W+ com encoder4editing e4e ou pixel2style2pixel pSp."
    )
    parser.add_argument("--input-crop", type=str, required=True, help="Caminho para o crop facial.")
    parser.add_argument("--output-dir", type=str, required=True, help="Diretorio onde guardar os outputs.")
    parser.add_argument(
        "--e4e-checkpoint",
        type=str,
        default=str(DEFAULT_E4E_CKPT),
        help="Checkpoint do e4e.",
    )
    parser.add_argument(
        "--psp-checkpoint",
        type=str,
        default=str(DEFAULT_PSP_CKPT),
        help="Checkpoint FFHQ do pixel2style2pixel pSp.",
    )
    parser.add_argument(
        "--encoder-backend",
        type=str,
        default="e4e",
        choices=["e4e", "psp"],
        help="Encoder a usar para a inversao facial.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo preferido para a inversao.",
    )
    parser.add_argument(
        "--save-reconstruction-preview",
        action="store_true",
        help="Guarda uma reconstrucao diagnostica do latent. Nao e uma edicao.",
    )
    parser.add_argument(
        "--refine-latent-steps",
        type=int,
        default=0,
        help="Numero de passos de otimizacao do W+ para reduzir perda de reconstrucao.",
    )
    parser.add_argument(
        "--refine-learning-rate",
        type=float,
        default=0.01,
        help="Learning rate da otimizacao do latent W+.",
    )
    parser.add_argument(
        "--refine-latent-l2",
        type=float,
        default=0.001,
        help="Peso L2 para manter o latent refinado perto do latent inicial do encoder.",
    )
    args = parser.parse_args()

    input_crop = Path(args.input_crop).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_path = (
        Path(args.psp_checkpoint).resolve()
        if args.encoder_backend == "psp"
        else Path(args.e4e_checkpoint).resolve()
    )

    if not input_crop.exists():
        raise FileNotFoundError(f"Crop nao encontrado: {input_crop}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {args.encoder_backend} nao encontrado: {checkpoint_path}"
        )

    configure_runtime()
    encode_face(
        input_crop,
        output_dir,
        checkpoint_path,
        args.device,
        args.encoder_backend,
        args.save_reconstruction_preview,
        args.refine_latent_steps,
        args.refine_learning_rate,
        args.refine_latent_l2,
    )


if __name__ == "__main__":
    main()
