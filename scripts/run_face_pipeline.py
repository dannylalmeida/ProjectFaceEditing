from __future__ import annotations

from face_pipeline_utils import (
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    build_parsing_model,
    configure_runtime_warnings,
    detect_faces_and_crops,
    ensure_directories,
    load_required_modules,
    parse_face_crop,
    resolve_input_image_path,
    save_face_outputs,
    suppress_known_stderr_noise,
)


def main() -> None:
    configure_runtime_warnings()
    input_image = resolve_input_image_path()
    with suppress_known_stderr_noise():
        ensure_directories()
        cv2, np, torch, _, init_parsing_model, img2tensor, normalize = load_required_modules()
        _, faces, _ = detect_faces_and_crops(input_image)

        model = build_parsing_model(init_parsing_model)

        print(f"Imagem selecionada: {input_image}")
        print(f"Faces detetadas: {len(faces)}")
        for face in faces:
            face_name = face["name"]
            crop = face["crop"]

            mask = parse_face_crop(crop, model, cv2, np, torch, img2tensor, normalize)
            mask_gray = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask_color = cv2.applyColorMap(mask_gray, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(crop, 0.55, mask_color, 0.45, 0)

            save_face_outputs(face_name, crop, mask_gray, mask_color, overlay, cv2, output_dir=OUTPUT_DIR)
            print(
                f"[{face['index']}/{len(faces)}] {face_name}: "
                f"score={face['score']:.4f}, "
                f"bbox={face['bbox']}"
            )

        print(f"Resultados guardados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
