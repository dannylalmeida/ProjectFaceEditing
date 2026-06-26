from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from src.editors.local_recolor import infer_target_color
from src.segmentation.mask_utils import normalize_edit_region


@dataclass(frozen=True)
class EditStrategy:
    description: str
    edit_region: str
    target_regions: list[str]
    color: str | None
    use_local_recolor: bool
    use_styleclip: bool
    use_repaint: bool
    reason: str


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    normalized = f" {(text or '').lower()} "
    return any(f" {word} " in normalized for word in words)


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return ascii_text.lower().replace("-", " ")


def route_attribute(
    description: str,
    edit_region: str = "auto",
    use_local_recolor: bool = True,
    use_styleclip: bool = False,
    use_repaint: bool = False,
) -> EditStrategy:
    normalized = _normalize_text(description)
    requested_region = normalize_edit_region(edit_region)
    color = infer_target_color(description)

    region = requested_region
    if region == "auto":
        if _contains_any(normalized, ("iris", "irises")):
            region = "iris"
        elif color and _contains_any(normalized, ("eye", "eyes", "olho", "olhos")):
            region = "iris"
        elif _contains_any(normalized, ("smile", "smiling", "sorriso", "mouth", "boca", "lips", "labios")):
            region = "mouth"
        elif _contains_any(normalized, ("nose", "nariz")):
            region = "nose"
        elif _contains_any(normalized, ("older", "younger", "old", "young", "velho", "velha", "jovem", "age")):
            region = "face"
        else:
            region = "face"

    if region == "eyes" and color:
        region = "iris"

    target_regions = {
        "iris": ["olhos"],
        "eyes": ["olhos"],
        "mouth": ["boca"],
        "face": ["pele"],
        "eyebrows": ["sobrancelhas"],
        "nose": ["nariz"],
        "ears": ["orelhas"],
        "neck": ["pescoco"],
    }.get(region, ["pele"])

    local_supported = region == "iris" and color is not None
    resolved_local = bool(use_local_recolor and local_supported)
    resolved_styleclip = bool(use_styleclip and not resolved_local)
    reason = "local color recolor" if resolved_local else "styleclip/generative source" if resolved_styleclip else "mask/debug only"

    if region == "iris":
        resolved_repaint = False
    else:
        resolved_repaint = bool(use_repaint)

    return EditStrategy(
        description=description,
        edit_region=region,
        target_regions=target_regions,
        color=color,
        use_local_recolor=resolved_local,
        use_styleclip=resolved_styleclip,
        use_repaint=resolved_repaint,
        reason=reason,
    )
