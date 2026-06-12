from __future__ import annotations


def mask_to_float(mask_uint8, np):
    mask_float = mask_uint8.astype(np.float32) / 255.0
    if mask_float.ndim == 2:
        mask_float = mask_float[..., None]
    return np.clip(mask_float, 0.0, 1.0)


def blend_with_original(original_bgr, edited_bgr, mask_uint8, np):
    alpha = mask_to_float(mask_uint8, np)
    blended = (
        original_bgr.astype(np.float32) * (1.0 - alpha)
        + edited_bgr.astype(np.float32) * alpha
    )
    result = blended.clip(0, 255).astype(np.uint8)
    if mask_uint8.ndim == 2:
        result[mask_uint8 <= 0] = original_bgr[mask_uint8 <= 0]
    return result

