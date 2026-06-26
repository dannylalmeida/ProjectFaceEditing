from __future__ import annotations


def mask_to_float(mask_uint8, np):
    mask_float = mask_uint8.astype(np.float32) / 255.0
    # Keep feathering at the edge without blending the whole edited region into a soft ghost.
    mask_float = np.clip(mask_float / 0.55, 0.0, 1.0)
    if mask_float.ndim == 2:
        mask_float = mask_float[..., None]
    return mask_float


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


def seamless_blend_with_original(original_bgr, edited_bgr, mask_uint8, cv2, np, levels: int = 4):
    """
    Laplacian pyramid blending — eliminates color seams at the edit boundary.
    Low-frequency content blends over a wide soft mask; high-frequency detail
    blends over a tight mask. Result: no visible cut even with a hard pixel mask.
    """
    mask_f = mask_uint8.astype(np.float32) / 255.0

    orig_f = original_bgr.astype(np.float32)
    edit_f = edited_bgr.astype(np.float32)

    # Gaussian pyramids
    gp_orig = [orig_f]
    gp_edit = [edit_f]
    gp_mask = [mask_f]
    for _ in range(levels - 1):
        gp_orig.append(cv2.pyrDown(gp_orig[-1]))
        gp_edit.append(cv2.pyrDown(gp_edit[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    # Laplacian pyramids for image pair
    lp_orig = []
    lp_edit = []
    for i in range(levels - 1):
        h, w = gp_orig[i].shape[:2]
        up_o = cv2.pyrUp(gp_orig[i + 1], dstsize=(w, h))
        up_e = cv2.pyrUp(gp_edit[i + 1], dstsize=(w, h))
        lp_orig.append(gp_orig[i] - up_o)
        lp_edit.append(gp_edit[i] - up_e)
    lp_orig.append(gp_orig[-1])
    lp_edit.append(gp_edit[-1])

    # Blend each pyramid level using the corresponding Gaussian-pyramid mask level.
    # i=0 is the finest (full-res) level — keep the mask tight so texture edges stay sharp.
    # i=levels-1 is the coarsest — apply wide extra softening so colour seams are
    # absorbed at low frequencies before reconstruction.
    blended_lp = []
    for i in range(levels):
        h, w = lp_orig[i].shape[:2]
        m = cv2.resize(gp_mask[i], (w, h), interpolation=cv2.INTER_LINEAR)
        # Extra blur increases with level index: 0 → minimal, levels-1 → widest
        extra_sigma = max(1.0, w * 0.02) * (1.0 + i * 0.8)
        k = (int(extra_sigma * 5) | 1)
        m = cv2.GaussianBlur(m, (k, k), extra_sigma)
        m = m[..., None]
        blended_lp.append(lp_orig[i] * (1.0 - m) + lp_edit[i] * m)

    # Reconstruct from coarsest to finest
    result = blended_lp[-1]
    for i in range(levels - 2, -1, -1):
        h, w = blended_lp[i].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + blended_lp[i]

    out = result.clip(0, 255).astype(np.uint8)
    if mask_uint8.ndim == 2:
        out[mask_uint8 <= 0] = original_bgr[mask_uint8 <= 0]
    return out
