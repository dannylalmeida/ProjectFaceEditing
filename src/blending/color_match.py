from __future__ import annotations


def match_region_color_lab(source_bgr, reference_bgr, mask_uint8, cv2, np):
    if not np.any(mask_uint8):
        return source_bgr

    region = mask_uint8 > 0
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    matched_lab = source_lab.copy()

    for channel in range(3):
        source_values = source_lab[..., channel][region]
        reference_values = reference_lab[..., channel][region]
        source_std = float(source_values.std())
        reference_std = float(reference_values.std())
        if source_std < 1e-3:
            matched_values = source_values + (float(reference_values.mean()) - float(source_values.mean()))
        else:
            matched_values = (
                (source_values - float(source_values.mean()))
                * (reference_std / source_std)
                + float(reference_values.mean())
            )
        matched_lab[..., channel][region] = matched_values

    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    output = source_bgr.copy()
    output[region] = matched_bgr[region]
    return output


def preserve_original_texture_in_region(
    edited_bgr,
    original_bgr,
    mask_uint8,
    cv2,
    np,
    blur_radius: int = 9,
    texture_strength: float = 0.85,
):
    if not np.any(mask_uint8):
        return edited_bgr

    matched_bgr = match_region_color_lab(edited_bgr, original_bgr, mask_uint8, cv2, np)
    blur_size = max(3, int(blur_radius) * 2 + 1)
    original_low = cv2.GaussianBlur(original_bgr, (blur_size, blur_size), 0).astype(np.float32)
    original_detail = original_bgr.astype(np.float32) - original_low
    preserved = matched_bgr.astype(np.float32) + original_detail * float(max(0.0, min(texture_strength, 1.0)))
    preserved = np.clip(preserved, 0, 255).astype(np.uint8)

    soft_mask = cv2.GaussianBlur((mask_uint8 > 0).astype(np.uint8) * 255, (11, 11), 0)
    alpha = np.clip(soft_mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    return (
        preserved.astype(np.float32) * alpha
        + edited_bgr.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)
