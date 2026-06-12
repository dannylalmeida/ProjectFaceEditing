from __future__ import annotations

import subprocess
from pathlib import Path


def run_e4e(script_path: str | Path, input_image: str | Path, output_dir: str | Path, device: str = "auto"):
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-InputImage",
        str(input_image),
        "-OutputDir",
        str(output_dir),
        "-Device",
        device,
        "-EncoderBackend",
        "e4e",
        "-ReconstruirPreview",
    ]
    return subprocess.run(command, check=False)

