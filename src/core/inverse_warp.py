from __future__ import annotations


def inverse_warp_to_original(final_aligned_bgr, original_image_bgr, mask_uint8, bbox, alignment, cv2, np):
    image_h, image_w = original_image_bgr.shape[:2]
    from src.blending.alpha_blend import seamless_blend_with_original

    if alignment.get("enabled") and alignment.get("inverse_matrix"):
        inverse_matrix = np.array(alignment["inverse_matrix"], dtype=np.float32)
        warped_final = cv2.warpAffine(
            final_aligned_bgr, inverse_matrix, (image_w, image_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
        warped_mask = cv2.warpAffine(
            mask_uint8, inverse_matrix, (image_w, image_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Simple alpha blend: the warp is from the same image so there is no colour
        # mismatch at the boundary. Poisson/Laplacian over-smooth subtle geometric
        # displacements, erasing the intended warp. The feathered mask already
        # provides a smooth edge without any additional blending overhead.
        alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
        result = (warped_final.astype(np.float32) * alpha
                  + original_image_bgr.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        return result, warped_mask

    x1, y1, x2, y2 = [int(v) for v in bbox]
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    resized_final = cv2.resize(final_aligned_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    alpha = (resized_mask.astype(np.float32) / 255.0)[..., None]
    roi = original_image_bgr[y1:y2, x1:x2].astype(np.float32)
    blended_roi = (resized_final.astype(np.float32) * alpha + roi * (1.0 - alpha)).clip(0, 255).astype(np.uint8)

    output = original_image_bgr.copy()
    output[y1:y2, x1:x2] = blended_roi
    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = resized_mask
    return output, full_mask
