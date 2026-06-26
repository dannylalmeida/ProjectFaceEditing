from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.blending.alpha_blend import blend_with_original
from src.blending.color_match import preserve_original_texture_in_region, restore_local_detail_in_region
from src.core.alignment import draw_landmarks_overlay
from src.core.image_io import load_image, save_bgr_image, save_rgb_image
from src.core.inverse_warp import inverse_warp_to_original
from src.editors.local_geometry import (
    apply_mouth_geometry_with_landmarks,
    resolve_mouth_geometry_intent,
    resolve_nose_geometry_intent,
    shrink_masked_region_with_inpaint,
)
from src.editors.local_recolor import recolor_iris
from src.editors.repaint_runner import run_repaint_or_inpainting
from src.evaluation.debug_outputs import (
    build_difference_image,
    compute_diff_stats,
    save_overlay,
    save_parsing_debug,
    write_json,
    write_text_log,
)
from src.legacy import PROJECT_DIR, ensure_legacy_scripts_on_path
from src.pipeline.attribute_router import route_attribute
from src.segmentation.face_parsing import build_standard_masks, run_face_parsing
from src.segmentation.facemesh_region import (
    build_custom_mouth_mask,
    build_custom_nose_mask,
    draw_numbered_facemesh_overlay,
    get_mouth_anchor_points,
)
from src.segmentation.iris_mask import build_iris_mask
from src.segmentation.mask_utils import create_edit_mask, extract_mask_outline_contours, get_edit_mask_defaults, refine_mask


DECREASE_TERMS = (
    "smaller",
    "small",
    "thin",
    "thinner",
    "narrower",
    "less",
    "shorter",
    "decrease",
    "reduce",
    "reduced",
    "shrink",
    "menor",
    "menores",
    "pequeno",
    "pequena",
    "mais pequeno",
    "mais pequena",
    "fino",
    "fina",
    "mais fino",
    "mais fina",
    "afinar",
    "diminuir",
    "diminuido",
    "diminuida",
    "reduzir",
    "reduzido",
    "reduzida",
    "encolher",
    "encolhido",
    "encolhida",
)


_RUNTIME_CACHE: dict[str, object] = {}


def _remove_small_mask_components(mask_uint8, cv2, np, min_area: int = 48):
    if mask_uint8 is None or not np.any(mask_uint8):
        return mask_uint8
    hard = (mask_uint8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    if num_labels <= 2:
        return mask_uint8
    cleaned = np.zeros_like(mask_uint8)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            cleaned[labels == label] = 255
    return cleaned


def _requests_decrease(description: str) -> bool:
    normalized = f" {(description or '').lower().replace('-', ' ')} "
    return any(term in normalized for term in DECREASE_TERMS)


@dataclass
class HybridPipelineConfig:
    input_image: str | Path = "dataset"
    output_dir: str | Path = PROJECT_DIR / "outputs" / "hybrid_edit"
    crop_metadata: str | Path | None = None
    description: str = ""
    source_description: str = ""
    target_description: str = ""
    edit_region: str = "auto"
    use_face_parsing: bool = True
    use_local_recolor: bool = True
    use_styleclip: bool = False
    styleclip_edited_image: str | Path | None = None
    mask_dilation: int = -1
    mask_erosion: int = 0
    mask_blur: int = -1
    mask_threshold: int = 1
    use_repaint: bool = False
    repaint_steps: int = 20
    repaint_strength: float = 0.35
    repaint_backend: str = "opencv"
    debug: bool = False
    margin_scale: float = 0.15
    local_strength: float = 0.82


def _copy_existing_reconstructions(output_dir: Path) -> dict[str, str | None]:
    candidates = {
        "psp_reconstruction": [
            PROJECT_DIR / "outputs" / "retinaface_psp" / "02_psp_inversion" / "psp_reconstruction_preview.png",
            PROJECT_DIR / "outputs" / "reconstruction_check" / "psp" / "02_psp_inversion" / "psp_reconstruction_preview.png",
        ],
        "e4e_reconstruction": [
            PROJECT_DIR / "outputs" / "retinaface_e4e" / "02_e4e_inversion" / "e4e_reconstruction_preview.png",
            PROJECT_DIR / "outputs" / "reconstruction_check" / "e4e" / "02_e4e_inversion" / "e4e_reconstruction_preview.png",
        ],
    }
    copied: dict[str, str | None] = {}
    for name, paths in candidates.items():
        destination = output_dir / f"{name}.png"
        copied[name] = None
        for path in paths:
            if path.exists():
                shutil.copyfile(path, destination)
                copied[name] = str(destination)
                break
    return copied


def _save_crop_metadata(path: Path, input_path: Path, primary_face: dict[str, object]) -> None:
    payload = {
        "input_image": str(input_path),
        "face_name": primary_face.get("name"),
        "score": primary_face.get("score"),
        "priority": primary_face.get("priority"),
        "bbox": list(primary_face.get("crop_bbox") or primary_face.get("bbox")),
        "detected_bbox": list(primary_face.get("detected_bbox")),
        "crop_bbox": list(primary_face.get("crop_bbox") or primary_face.get("bbox")),
        "landmarks": primary_face.get("landmarks", {}),
        "alignment": primary_face.get("alignment", {"enabled": False}),
        "crop_path": str(path.parent / "aligned_original.png"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _get_runtime(load_required_modules, build_parsing_model_func, init_parsing: bool):
    runtime = _RUNTIME_CACHE.get("modules")
    if runtime is None:
        runtime = load_required_modules()
        _RUNTIME_CACHE["modules"] = runtime
    cv2, np, torch, _, init_parsing_model, img2tensor, normalize = runtime
    model = None
    if init_parsing:
        model = _RUNTIME_CACHE.get("parsing_model")
        if model is None:
            model = build_parsing_model_func(init_parsing_model)
            _RUNTIME_CACHE["parsing_model"] = model
    return cv2, np, torch, init_parsing_model, img2tensor, normalize, model


def run_hybrid_pipeline(config: HybridPipelineConfig) -> dict[str, object]:
    ensure_legacy_scripts_on_path()
    from face_pipeline_utils import (
        build_parsing_model,
        configure_runtime_warnings,
        detect_faces_and_crops,
        ensure_directories,
        load_required_modules,
        resolve_input_image_candidates,
        suppress_known_stderr_noise,
    )

    output_dir = Path(config.output_dir).resolve()
    configure_runtime_warnings()
    crop_metadata = None
    crop_metadata_path = Path(config.crop_metadata).resolve() if config.crop_metadata else None
    if crop_metadata_path:
        if not crop_metadata_path.exists():
            raise FileNotFoundError(f"Crop metadata not found: {crop_metadata_path}")
        crop_metadata = json.loads(crop_metadata_path.read_text(encoding="utf-8"))
        input_path = Path(crop_metadata["input_image"]).resolve()
        input_candidates = [input_path]
    else:
        input_candidates = resolve_input_image_candidates(config.input_image, max_attempts=25)
        input_path = input_candidates[0]
    _reset_output_dir(output_dir)
    ensure_directories(output_dir=output_dir)

    description = config.target_description or config.description
    strategy = route_attribute(
        description,
        edit_region=config.edit_region,
        use_local_recolor=config.use_local_recolor,
        use_styleclip=config.use_styleclip,
        use_repaint=config.use_repaint,
    )

    with suppress_known_stderr_noise():
        cv2, np, torch, _, img2tensor, normalize, cached_parsing_model = _get_runtime(
            load_required_modules,
            build_parsing_model,
            config.use_face_parsing,
        )
        if crop_metadata is not None:
            original_bgr = load_image(input_path, cv2)
            crop_path = Path(crop_metadata["crop_path"]).resolve()
            aligned_original_bgr = load_image(crop_path, cv2)
            primary_face = {
                "name": crop_metadata.get("face_name", "primary_face"),
                "score": crop_metadata.get("score"),
                "priority": crop_metadata.get("priority"),
                "bbox": crop_metadata.get("bbox"),
                "crop_bbox": crop_metadata.get("crop_bbox") or crop_metadata.get("bbox"),
                "detected_bbox": crop_metadata.get("detected_bbox"),
                "landmarks": crop_metadata.get("landmarks", {}),
                "alignment": crop_metadata.get("alignment") or {"enabled": False},
                "crop": aligned_original_bgr,
            }
        else:
            detection_errors = []
            for candidate_path in input_candidates:
                try:
                    original_bgr, faces, _ = detect_faces_and_crops(candidate_path, margin_scale=config.margin_scale)
                    input_path = candidate_path
                    break
                except Exception as exc:
                    detection_errors.append(f"{candidate_path}: {type(exc).__name__}: {exc}")
            else:
                details = "\n".join(detection_errors[:8])
                raise ValueError(
                    "Nao consegui encontrar uma imagem valida para editar nas tentativas aleatorias. "
                    "Isto costuma acontecer quando a imagem nao tem uma face frontal detetavel.\n"
                    f"Tentativas falhadas:\n{details}"
                )
            primary_face = faces[0]
            aligned_original_bgr = primary_face["crop"]
        model = cached_parsing_model if config.use_face_parsing else None
        parsing_map = (
            run_face_parsing(aligned_original_bgr, cv2, np, torch, img2tensor, normalize, model)
            if model is not None
            else None
        )

        if config.styleclip_edited_image:
            styleclip_edit_bgr = load_image(config.styleclip_edited_image, cv2)
            crop_h, crop_w = aligned_original_bgr.shape[:2]
            styleclip_edit_bgr = cv2.resize(styleclip_edit_bgr, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        else:
            styleclip_edit_bgr = aligned_original_bgr.copy()

        nose_keepout_mask = None
        mouth_anchor_points = None
        mouth_keepout_mask = None
        edit_region_mask_source = "face_parsing"
        if parsing_map is None:
            selected_mask = np.ones(aligned_original_bgr.shape[:2], dtype=np.uint8) * 255
            masks = {"selected_mask": selected_mask}
            labels = []
            edit_region_mask_source = "full_image_fallback"
            if strategy.edit_region == "nose":
                custom_nose_mask = build_custom_nose_mask(aligned_original_bgr, cv2, np)
                if custom_nose_mask is not None:
                    selected_mask = custom_nose_mask
                    masks["nose_mask"] = custom_nose_mask
                    edit_region_mask_source = "facemesh_nose"
            elif strategy.edit_region == "mouth":
                custom_mouth_mask = build_custom_mouth_mask(aligned_original_bgr, cv2, np)
                mouth_anchor_points = get_mouth_anchor_points(aligned_original_bgr, cv2)
                if custom_mouth_mask is not None:
                    selected_mask = custom_mouth_mask
                    masks["mouth_mask"] = custom_mouth_mask
                    edit_region_mask_source = "facemesh_mouth"
        else:
            masks = build_standard_masks(parsing_map, cv2, np)
            selected_mask, labels = create_edit_mask(strategy.edit_region, parsing_map, cv2, np, dilation=0)
            if strategy.edit_region == "iris":
                iris_mask = build_iris_mask(aligned_original_bgr, masks["eyes_mask"], cv2, np)
                masks["iris_mask"] = iris_mask
                selected_mask = iris_mask
            elif strategy.edit_region == "nose":
                custom_nose_mask = build_custom_nose_mask(aligned_original_bgr, cv2, np)
                if custom_nose_mask is not None:
                    mouth_keepout = cv2.dilate(
                        masks["mouth_mask"],
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                        iterations=1,
                    )
                    nose_keepout_mask = mouth_keepout
                    custom_nose_mask = cv2.bitwise_and(custom_nose_mask, cv2.bitwise_not(mouth_keepout))
                    selected_mask = custom_nose_mask
                    masks["nose_mask"] = custom_nose_mask
                    edit_region_mask_source = "facemesh_nose"
            elif strategy.edit_region == "mouth":
                custom_mouth_mask = build_custom_mouth_mask(aligned_original_bgr, cv2, np)
                mouth_anchor_points = get_mouth_anchor_points(aligned_original_bgr, cv2)
                if custom_mouth_mask is not None:
                    mouth_support = cv2.max(masks["mouth_mask"], masks["teeth_mask"])
                    mouth_support = cv2.dilate(
                        mouth_support,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 11)),
                        iterations=1,
                    )
                    selected_mask = cv2.bitwise_and(custom_mouth_mask, mouth_support)
                    if not np.any(selected_mask):
                        selected_mask = custom_mouth_mask
                    masks["mouth_mask"] = custom_mouth_mask
                    mouth_keepout_mask = cv2.dilate(
                        masks["teeth_mask"],
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)),
                        iterations=1,
                    )
                    edit_region_mask_source = "facemesh_mouth"
            else:
                masks["iris_mask"] = np.zeros_like(selected_mask)
                nose_keepout_mask = None
                mouth_anchor_points = None
                mouth_keepout_mask = None

        defaults = get_edit_mask_defaults(strategy.edit_region)
        mask_dilation = config.mask_dilation if config.mask_dilation >= 0 else defaults["dilation"]
        mask_blur = config.mask_blur if config.mask_blur >= 0 else defaults["blur"]
        selected_mask_hard = refine_mask(
            selected_mask,
            dilation=mask_dilation,
            erosion=config.mask_erosion,
            blur=0,
            threshold=config.mask_threshold,
            cv2=cv2,
            np=np,
        )

        edited_source_bgr = styleclip_edit_bgr.copy()
        direct_refinement = {"mode": "none"}
        if strategy.use_local_recolor:
            if strategy.edit_region == "iris" and strategy.color:
                edited_source_bgr = recolor_iris(
                    aligned_original_bgr,
                    selected_mask,
                    strategy.color,
                    strength=1.0,
                    cv2=cv2,
                    np=np,
                    blur=1,
                )
                direct_refinement = {"mode": "local_iris_recolor", "color": strategy.color}

        geometry_mask = None
        nose_geometry_intent = resolve_nose_geometry_intent(description) if strategy.edit_region == "nose" else None
        if strategy.edit_region == "nose" and nose_geometry_intent is not None:
            edited_source_bgr, geometry_mask = shrink_masked_region_with_inpaint(
                aligned_original_bgr,
                selected_mask_hard,
                cv2,
                np,
                scale_x=nose_geometry_intent.scale_x,
                scale_y=nose_geometry_intent.scale_y,
                feather=5,
                inpaint_radius=3,
                center_y_ratio=nose_geometry_intent.center_y_ratio,
                lower_extension_ratio=nose_geometry_intent.lower_extension_ratio,
                keepout_mask=nose_keepout_mask,
            )
            direct_refinement = {
                "mode": "direct_nose_smooth_warp",
                "scale_x": nose_geometry_intent.scale_x,
                "scale_y": nose_geometry_intent.scale_y,
                "center_y_ratio": nose_geometry_intent.center_y_ratio,
                "lower_extension_ratio": nose_geometry_intent.lower_extension_ratio,
                "intent": nose_geometry_intent.reason,
                "uses_inpaint_background": True,
                "styleclip_source_discarded_for_geometry": bool(config.use_styleclip),
            }
        mouth_geometry_intent = resolve_mouth_geometry_intent(description) if strategy.edit_region == "mouth" else None
        if strategy.edit_region == "mouth" and mouth_geometry_intent is not None:
            edited_source_bgr, geometry_mask = apply_mouth_geometry_with_landmarks(
                aligned_original_bgr,
                selected_mask_hard,
                cv2,
                np,
                anchor_points=mouth_anchor_points,
                scale_x=mouth_geometry_intent.scale_x,
                scale_y=mouth_geometry_intent.scale_y,
                smile_strength=mouth_geometry_intent.smile_strength,
                feather=mouth_geometry_intent.feather,
                inpaint_radius=3,
                keepout_mask=mouth_keepout_mask,
                corner_lift=mouth_geometry_intent.corner_lift,
                uniform_lip_height=mouth_geometry_intent.uniform_lip_height,
            )
            direct_refinement = {
                "mode": "direct_mouth_smooth_warp",
                "scale_x": mouth_geometry_intent.scale_x,
                "scale_y": mouth_geometry_intent.scale_y,
                "smile_strength": mouth_geometry_intent.smile_strength,
                "corner_lift": mouth_geometry_intent.corner_lift,
                "uniform_lip_height": mouth_geometry_intent.uniform_lip_height,
                "intent": mouth_geometry_intent.reason,
                "uses_inpaint_background": False,
                "styleclip_source_discarded_for_geometry": bool(config.use_styleclip),
            }
            if mouth_keepout_mask is not None and np.any(mouth_keepout_mask):
                direct_refinement["inner_mouth_keepout"] = True

        if geometry_mask is not None:
            final_mask_aligned = geometry_mask
        else:
            final_mask_aligned = refine_mask(
                selected_mask_hard,
                dilation=0,
                erosion=0,
                blur=mask_blur,
                threshold=config.mask_threshold,
                cv2=cv2,
                np=np,
            )
        mouth_keepout_hard = None
        mouth_geometry_active = strategy.edit_region == "mouth" and mouth_geometry_intent is not None
        mouth_corner_geometry_active = bool(
            mouth_geometry_active and getattr(mouth_geometry_intent, "corner_lift", 0.0) > 0.0
        )
        if strategy.edit_region == "mouth" and mouth_keepout_mask is not None and np.any(mouth_keepout_mask):
            if mouth_corner_geometry_active:
                direct_refinement["inner_mouth_guarded_in_corner_warp"] = True
            else:
                mouth_keepout_hard = (mouth_keepout_mask > 0).astype(np.uint8) * 255
                edited_source_bgr[mouth_keepout_hard > 0] = aligned_original_bgr[mouth_keepout_hard > 0]
                if mouth_geometry_active:
                    direct_refinement["inner_mouth_restored_inside_final_mask"] = True
                else:
                    final_mask_aligned = cv2.bitwise_and(final_mask_aligned, cv2.bitwise_not(mouth_keepout_hard))
                    direct_refinement["inner_mouth_removed_from_final_mask"] = True
        if strategy.edit_region == "mouth" and mouth_geometry_intent is not None:
            final_mask_aligned = _remove_small_mask_components(final_mask_aligned, cv2, np, min_area=24)
        if strategy.edit_region == "nose" and geometry_mask is None:
            edited_source_bgr = preserve_original_texture_in_region(
                edited_source_bgr,
                aligned_original_bgr,
                final_mask_aligned,
                cv2,
                np,
            )
        if strategy.edit_region == "mouth" and mouth_geometry_intent is not None:
            # apply_mouth_geometry_with_landmarks already composites warp with original
            # via the feathered change_mask. blend_with_original would create a second
            # blend pass, weakening edge transitions non-uniformly and producing seam artifacts.
            final_blended_aligned_bgr = edited_source_bgr
            direct_refinement["aligned_blend_skipped"] = "geometry_already_composited"
        else:
            final_blended_aligned_bgr = blend_with_original(aligned_original_bgr, edited_source_bgr, final_mask_aligned, np)
        if mouth_keepout_hard is not None:
            final_blended_aligned_bgr[mouth_keepout_hard > 0] = aligned_original_bgr[mouth_keepout_hard > 0]
        effective_use_repaint = bool(strategy.use_repaint)
        if strategy.edit_region == "mouth" and mouth_geometry_intent is not None:
            effective_use_repaint = False
            if strategy.use_repaint:
                direct_refinement["repaint_skipped_for_landmark_geometry"] = True
        final_repainted_aligned_bgr, repaint_edge_mask, repaint_info = run_repaint_or_inpainting(
            final_blended_aligned_bgr,
            final_mask_aligned,
            cv2,
            np,
            steps=config.repaint_steps if effective_use_repaint else 0,
            strength=config.repaint_strength if effective_use_repaint else 0.0,
            backend=config.repaint_backend,
            work_dir=output_dir,
            project_dir=PROJECT_DIR,
            conda_env="styleclip",
            device="auto",
        )
        final_aligned_for_warp = final_repainted_aligned_bgr if effective_use_repaint else final_blended_aligned_bgr
        if mouth_keepout_hard is not None:
            final_aligned_for_warp[mouth_keepout_hard > 0] = aligned_original_bgr[mouth_keepout_hard > 0]
        final_mask_for_warp = final_mask_aligned
        if effective_use_repaint and np.any(repaint_edge_mask):
            final_mask_for_warp = cv2.max(final_mask_aligned, repaint_edge_mask)
        aligned_touched = cv2.absdiff(aligned_original_bgr, final_aligned_for_warp)
        aligned_touched_mask = (aligned_touched.max(axis=2) > 0).astype(np.uint8) * 255
        final_mask_for_warp = cv2.max(final_mask_for_warp, aligned_touched_mask)
        if mouth_keepout_hard is not None and not mouth_geometry_active:
            final_mask_for_warp = cv2.bitwise_and(final_mask_for_warp, cv2.bitwise_not(mouth_keepout_hard))
        if strategy.edit_region == "mouth" and mouth_geometry_intent is not None:
            final_mask_for_warp = _remove_small_mask_components(final_mask_for_warp, cv2, np, min_area=24)
        editable_region_outline_aligned = extract_mask_outline_contours(final_mask_for_warp, cv2, np)

        final_on_original_bgr, full_mask_on_original = inverse_warp_to_original(
            final_aligned_for_warp,
            original_bgr,
            final_mask_for_warp,
            primary_face.get("crop_bbox") or primary_face.get("bbox"),
            primary_face.get("alignment") or {"enabled": False},
            cv2,
            np,
        )
        # Sharpening a geometry-warp result creates ringing at the edit boundary Ã¢â‚¬â€ skipped for mouth.
        editable_region_outline_full = extract_mask_outline_contours(full_mask_on_original, cv2, np)

        landmarks_overlay_bgr = draw_landmarks_overlay(original_bgr, primary_face.get("landmarks", {}), cv2)
        selected_overlay_bgr = cv2.imread(str(save_overlay(output_dir / "selected_mask_overlay.png", aligned_original_bgr, final_mask_for_warp, cv2, np)))
        iris_overlay_bgr = None
        if config.debug:
            if np.any(masks.get("iris_mask", np.zeros_like(selected_mask_hard))):
                iris_overlay_bgr = cv2.imread(str(save_overlay(output_dir / "iris_overlay.png", aligned_original_bgr, masks["iris_mask"], cv2, np, color_bgr=(255, 80, 0), alpha=0.55)))

    paths: dict[str, str | None] = {}
    paths["edit_mask"] = str(save_rgb_image(output_dir / "edit_mask.png", final_mask_for_warp))
    if (output_dir / "selected_mask_overlay.png").exists():
        paths["selected_mask_overlay"] = str(output_dir / "selected_mask_overlay.png")
    if (output_dir / "iris_overlay.png").exists():
        paths["iris_overlay"] = str(output_dir / "iris_overlay.png")
    facemesh_overlay_bgr = draw_numbered_facemesh_overlay(original_bgr, cv2, scale=2.0)
    if facemesh_overlay_bgr is not None:
        paths["landmarks"] = str(save_bgr_image(output_dir / "landmarks_478_enumerados.png", facemesh_overlay_bgr, cv2))
    paths["original_image"] = str(save_bgr_image(output_dir / "original_image.png", original_bgr, cv2))
    paths["final_on_original"] = str(save_bgr_image(output_dir / "final_on_original.png", final_on_original_bgr, cv2))
    paths["resultado_final"] = str(save_bgr_image(output_dir / "resultado_final.png", final_on_original_bgr, cv2))

    if config.debug:
        paths["aligned_original"] = str(save_bgr_image(output_dir / "aligned_original.png", aligned_original_bgr, cv2))
        paths["landmarks_overlay"] = str(save_bgr_image(output_dir / "landmarks_overlay.png", landmarks_overlay_bgr, cv2))
        paths["crop_debug"] = str(save_bgr_image(output_dir / "crop_debug.png", aligned_original_bgr, cv2))
        paths["styleclip_edit"] = str(save_bgr_image(output_dir / "styleclip_edit.png", styleclip_edit_bgr, cv2))
        paths["selected_mask"] = str(save_rgb_image(output_dir / "selected_mask.png", selected_mask_hard))
        paths["selected_edit_mask"] = str(save_rgb_image(output_dir / "selected_edit_mask.png", selected_mask_hard))
        paths["final_edit_mask"] = str(save_rgb_image(output_dir / "final_edit_mask.png", final_mask_for_warp))
        paths["final_blended_aligned"] = str(save_bgr_image(output_dir / "final_blended_aligned.png", final_blended_aligned_bgr, cv2))
        paths["final_repainted_aligned"] = str(save_bgr_image(output_dir / "final_repainted_aligned.png", final_aligned_for_warp, cv2))
        paths["inverse_warp_mask"] = str(save_rgb_image(output_dir / "inverse_warp_mask.png", full_mask_on_original))
        paths["edit_mask_on_original"] = str(save_rgb_image(output_dir / "edit_mask_on_original.png", full_mask_on_original))
        paths["localized_crop"] = str(save_bgr_image(output_dir / "localized_crop.png", final_blended_aligned_bgr, cv2))
        paths["localized_on_image"] = str(save_bgr_image(output_dir / "localized_on_image.png", final_on_original_bgr, cv2))
        paths.update(_copy_existing_reconstructions(output_dir))

    if config.debug and parsing_map is not None:
        masks_to_save = {**masks, "selected_mask": selected_mask_hard, "selected_edit_mask": selected_mask_hard, "final_edit_mask": final_mask_for_warp}
        paths.update(save_parsing_debug(output_dir, parsing_map, masks_to_save, cv2, np))

    validation_report = {
        "aligned_diff": compute_diff_stats(aligned_original_bgr, final_aligned_for_warp, final_mask_for_warp, np),
        "full_image_diff": compute_diff_stats(original_bgr, final_on_original_bgr, full_mask_on_original, np),
        "outside_mask_should_be_unchanged": True,
        "mask_pixels_aligned": int((final_mask_for_warp > 0).sum()),
        "mask_pixels_full": int((full_mask_on_original > 0).sum()),
        "selected_attribute": strategy.edit_region,
        "editable_region_outline_contours": len(editable_region_outline_aligned),
        "target_regions": strategy.target_regions,
        "local_recolor_used": strategy.use_local_recolor,
        "styleclip_used": config.use_styleclip,
        "repaint_used": effective_use_repaint,
    }

    if config.debug:
        paths["difference_outside_mask"] = str(
            save_bgr_image(
                output_dir / "difference_outside_mask.png",
                build_difference_image(original_bgr, final_on_original_bgr, full_mask_on_original, cv2, np, outside=True),
                cv2,
            )
        )
        paths["difference_inside_mask"] = str(
            save_bgr_image(
                output_dir / "difference_inside_mask.png",
                build_difference_image(original_bgr, final_on_original_bgr, full_mask_on_original, cv2, np, outside=False),
                cv2,
            )
        )
        paths["repaint_input"] = str(save_bgr_image(output_dir / "repaint_input.png", final_blended_aligned_bgr, cv2))
        paths["repaint_mask"] = str(save_rgb_image(output_dir / "repaint_mask.png", repaint_edge_mask if effective_use_repaint else final_mask_for_warp))
        paths["repaint_output"] = str(save_bgr_image(output_dir / "repaint_output.png", final_aligned_for_warp, cv2))

    crop_metadata_path = output_dir / "primary_face.json"
    if config.debug:
        _save_crop_metadata(crop_metadata_path, input_path, primary_face)
        paths["crop_metadata"] = str(crop_metadata_path)

    params_log = {
        "command": "hybrid_local_edit",
        "input_image": input_path,
        "description": description,
        "source_description": config.source_description,
        "target_description": config.target_description,
        "edit_region": config.edit_region,
        "selected_attribute": strategy.edit_region,
        "selected_color": strategy.color,
        "strategy": strategy.reason,
        "use_face_parsing": config.use_face_parsing,
        "use_local_recolor": strategy.use_local_recolor,
        "use_styleclip": config.use_styleclip,
        "use_repaint_requested": strategy.use_repaint,
        "use_repaint": effective_use_repaint,
        "mask_dilation": mask_dilation,
        "mask_erosion": config.mask_erosion,
        "mask_blur": mask_blur,
        "mask_threshold": config.mask_threshold,
        "repaint_steps": config.repaint_steps,
        "repaint_strength": config.repaint_strength,
        "repaint_backend": config.repaint_backend,
        "repaint_info": repaint_info,
        "direct_refinement": direct_refinement,
    }
    if config.debug:
        write_text_log(output_dir / "params_log.txt", params_log)
        (output_dir / "latent_shape_log.txt").write_text("latent: not used by local hybrid edit\n", encoding="utf-8")
    write_json(output_dir / "validation_report.json", validation_report)

    editable_region_outline = {
        "region": strategy.edit_region,
        "source": edit_region_mask_source,
        "coordinate_space": "aligned_crop",
        "contours": editable_region_outline_aligned,
        "full_image_coordinate_space": "original_image",
        "full_image_contours": editable_region_outline_full,
        "mask_path": paths.get("edit_mask"),
        "overlay_path": paths.get("selected_mask_overlay"),
    }

    metadata = {
        "operation": "hybrid_face_edit",
        "rule": "Original aligned image is the final base; generated/recolored outputs are sources only.",
        "input_image": str(input_path),
        "output_dir": str(output_dir),
        "strategy": strategy.__dict__,
        "region_labels": labels,
        "direct_refinement": direct_refinement,
        "repaint_info": repaint_info,
        "paths": paths,
        "editable_region_outline": editable_region_outline,
        "validation_report": validation_report,
        "alignment": primary_face.get("alignment") or {"enabled": False},
        "bbox": primary_face.get("crop_bbox") or primary_face.get("bbox"),
    }
    if config.debug:
        write_json(output_dir / "hybrid_edit_metadata.json", metadata)
    write_json(output_dir / "localized_edit_metadata.json", metadata)
    return metadata
