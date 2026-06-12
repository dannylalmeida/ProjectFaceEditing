from __future__ import annotations

from src.blending.alpha_blend import blend_with_original


def laplacian_blend(original_bgr, edited_bgr, mask_uint8, cv2, np, levels: int = 4):
    if levels <= 1 or not np.any(mask_uint8):
        return blend_with_original(original_bgr, edited_bgr, mask_uint8, np)

    mask = (mask_uint8.astype(np.float32) / 255.0)[..., None]
    gaussian_original = [original_bgr.astype(np.float32)]
    gaussian_edited = [edited_bgr.astype(np.float32)]
    gaussian_mask = [mask]
    for _ in range(levels - 1):
        gaussian_original.append(cv2.pyrDown(gaussian_original[-1]))
        gaussian_edited.append(cv2.pyrDown(gaussian_edited[-1]))
        gaussian_mask.append(cv2.pyrDown(gaussian_mask[-1]))

    lap_original = [gaussian_original[-1]]
    lap_edited = [gaussian_edited[-1]]
    for i in range(levels - 1, 0, -1):
        size = (gaussian_original[i - 1].shape[1], gaussian_original[i - 1].shape[0])
        lap_original.append(gaussian_original[i - 1] - cv2.pyrUp(gaussian_original[i], dstsize=size))
        lap_edited.append(gaussian_edited[i - 1] - cv2.pyrUp(gaussian_edited[i], dstsize=size))

    blended = []
    for lo, le, gm in zip(lap_original, lap_edited, reversed(gaussian_mask)):
        if gm.ndim == 2:
            gm = gm[..., None]
        blended.append(lo * (1.0 - gm) + le * gm)

    result = blended[0]
    for level in range(1, len(blended)):
        size = (blended[level].shape[1], blended[level].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended[level]
    return result.clip(0, 255).astype(np.uint8)

