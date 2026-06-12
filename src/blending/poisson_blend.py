from __future__ import annotations


def poisson_blend(original_bgr, edited_bgr, mask_uint8, cv2, np):
    if not np.any(mask_uint8):
        return original_bgr
    ys, xs = np.where(mask_uint8 > 0)
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    hard_mask = (mask_uint8 > 0).astype(np.uint8) * 255
    return cv2.seamlessClone(edited_bgr, original_bgr, hard_mask, center, cv2.NORMAL_CLONE)

