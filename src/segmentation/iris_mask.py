from __future__ import annotations


def _component_center_from_candidates(component_mask, candidates, stats_row, cv2, np):
    candidate = cv2.bitwise_and(component_mask, candidates)
    moments = cv2.moments(candidate)
    if moments["m00"] > 0:
        return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
    x, y, w, h, _ = stats_row
    return int(x + w / 2.0), int(y + h / 2.0)


def build_iris_mask(aligned_bgr, eyes_mask_uint8, cv2, np):
    if not np.any(eyes_mask_uint8):
        return np.zeros_like(eyes_mask_uint8)

    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    iris_candidates = (((gray > 18) & (gray < 190) & (saturation > 8)) | (gray < 85)).astype(np.uint8) * 255

    iris_mask = np.zeros_like(eyes_mask_uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((eyes_mask_uint8 > 0).astype(np.uint8), 8)
    for component_idx in range(1, component_count):
        x, y, w, h, area = stats[component_idx]
        if area < 20 or w < 4 or h < 4:
            continue

        component_mask = ((labels == component_idx).astype(np.uint8) * 255)
        center = _component_center_from_candidates(component_mask, iris_candidates, stats[component_idx], cv2, np)
        axes = (
            max(3, int(round(w * 0.30))),
            max(3, int(round(h * 0.42))),
        )
        ellipse = np.zeros_like(eyes_mask_uint8)
        cv2.ellipse(ellipse, center, axes, 0, 0, 360, 255, -1)
        ellipse = cv2.bitwise_and(ellipse, component_mask)

        filtered = cv2.bitwise_and(ellipse, iris_candidates)
        if np.any(filtered):
            filtered = cv2.morphologyEx(
                filtered,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            filtered = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            filtered = cv2.bitwise_and(filtered, ellipse)
            iris_mask = cv2.max(iris_mask, filtered)
        else:
            iris_mask = cv2.max(iris_mask, ellipse)

    if not np.any(iris_mask):
        eroded = cv2.erode(eyes_mask_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        return eroded

    return iris_mask

