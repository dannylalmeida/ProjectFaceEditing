from __future__ import annotations


NOSE_MASK_PADDING = 4


def _landmark_to_pixel(landmark, width: int, height: int, scale: float = 1.0) -> tuple[int, int]:
    x = min(max(float(landmark.x), 0.0), 1.0)
    y = min(max(float(landmark.y), 0.0), 1.0)
    return int(round(x * (width - 1) * scale)), int(round(y * (height - 1) * scale))


def _detect_facemesh_landmarks(image_bgr, cv2):
    import mediapipe as mp

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    results = face_mesh.process(image_rgb)
    face_mesh.close()
    if not results.multi_face_landmarks:
        return None, mp
    return results.multi_face_landmarks[0].landmark, mp


def detect_facemesh_landmarks(image_bgr, cv2):
    return _detect_facemesh_landmarks(image_bgr, cv2)


def build_facemesh_polygon_mask(image_bgr, landmark_indices: list[int], cv2, np):
    height, width = image_bgr.shape[:2]
    landmarks, _ = _detect_facemesh_landmarks(image_bgr, cv2)
    if landmarks is None:
        return None

    points = []
    for index in landmark_indices:
        if index >= len(landmarks):
            return None
        points.append(_landmark_to_pixel(landmarks[index], width, height))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def build_custom_nose_mask(image_bgr, cv2, np, landmarks=None, mp=None):
    height, width = image_bgr.shape[:2]
    if landmarks is None or mp is None:
        landmarks, mp = _detect_facemesh_landmarks(image_bgr, cv2)
    if landmarks is None:
        return None

    nose_indices = sorted({index for edge in mp.solutions.face_mesh.FACEMESH_NOSE for index in edge})
    points = [_landmark_to_pixel(landmarks[index], width, height) for index in nose_indices]

    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if NOSE_MASK_PADDING > 0:
        kernel_size = NOSE_MASK_PADDING * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def draw_numbered_facemesh_overlay(image_bgr, cv2, scale: float = 2.0, landmarks=None, mp=None):
    if landmarks is None or mp is None:
        landmarks, mp = _detect_facemesh_landmarks(image_bgr, cv2)
    if landmarks is None:
        return None

    height, width = image_bgr.shape[:2]
    if scale != 1.0:
        overlay = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        overlay = image_bgr.copy()

    mesh = mp.solutions.face_mesh
    line_thickness = max(1, int(round(scale)))
    contour_thickness = max(2, int(round(2 * scale)))
    point_radius = max(1, int(round(1.6 * scale)))
    font_scale = max(0.20, 0.22 * scale)
    text_thickness = max(1, int(round(scale)))
    outline_thickness = text_thickness + max(1, int(round(scale)))
    label_offset = max(2, int(round(2.5 * scale)))

    for connections, color, thickness in (
        (mesh.FACEMESH_TESSELATION, (95, 95, 95), line_thickness),
        (mesh.FACEMESH_CONTOURS, (245, 245, 245), contour_thickness),
        (mesh.FACEMESH_IRISES, (0, 255, 0), contour_thickness),
    ):
        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            start = _landmark_to_pixel(landmarks[start_idx], width, height, scale)
            end = _landmark_to_pixel(landmarks[end_idx], width, height, scale)
            cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)

    for index, landmark in enumerate(landmarks):
        x, y = _landmark_to_pixel(landmark, width, height, scale)
        label_pos = (x + label_offset, y - label_offset)
        cv2.circle(overlay, (x, y), point_radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), point_radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            str(index),
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return overlay
