from __future__ import annotations


def inverse_warp_to_original(final_aligned_bgr, original_image_bgr, mask_uint8, bbox, alignment, cv2, np):
    image_h, image_w = original_image_bgr.shape[:2]
    if alignment.get("enabled") and alignment.get("inverse_matrix"):
        inverse_matrix = np.array(alignment["inverse_matrix"], dtype=np.float32)
        warped_final = cv2.warpAffine(
            final_aligned_bgr,
            inverse_matrix,
            (image_w, image_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.warpAffine(
            mask_uint8,
            inverse_matrix,
            (image_w, image_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        from src.blending.alpha_blend import blend_with_original

        return blend_with_original(original_image_bgr, warped_final, warped_mask, np), warped_mask

    x1, y1, x2, y2 = [int(v) for v in bbox]
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    resized_final = cv2.resize(final_aligned_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    from src.blending.alpha_blend import blend_with_original

    output = original_image_bgr.copy()
    roi = original_image_bgr[y1:y2, x1:x2]
    output[y1:y2, x1:x2] = blend_with_original(roi, resized_final, resized_mask, np)
    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = resized_mask
    return output, full_mask

