from __future__ import annotations

import subprocess
from pathlib import Path


def run_styleclip(
    script_path: str | Path,
    latent_path: str | Path,
    target_description: str,
    output_dir: str | Path,
    source_description: str = "auto",
    steps: int = 5,
    edit_strength: float = 0.04,
    learning_rate: float = 0.02,
    layers_start: int = 8,
    layers_end: int = 17,
    lambda_latent: float = 0.05,
    delta_clamp: float = 0.08,
):
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-LatentPath",
        str(latent_path),
        "-TargetDescription",
        target_description,
        "-SourceDescription",
        source_description,
        "-OutputDir",
        str(output_dir),
        "-Step",
        str(steps),
        "-EditStrength",
        str(edit_strength),
        "-LearningRate",
        str(learning_rate),
        "-LatentLayerMin",
        str(layers_start),
        "-LatentLayerMax",
        str(layers_end),
        "-L2Lambda",
        str(lambda_latent),
        "-MaxLatentDelta",
        str(delta_clamp),
    ]
    return subprocess.run(command, check=False)

