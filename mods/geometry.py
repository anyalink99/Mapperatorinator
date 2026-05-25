"""Geometry post-processing mods that operate directly on .osu text.

These work on a finished beatmap (string or file) so they can be reused in the
inference pipeline, in standalone scripts, or on any external .osu file.
"""
from __future__ import annotations

from typing import Optional

PLAYFIELD = (512, 384)

# osu! hit-object type bitmask (4th comma field)
_CIRCLE = 1 << 0
_SLIDER = 1 << 1
_SPINNER = 1 << 3


def _iter_sections(lines: list[str]):
    """Yield (section_name, start_idx, end_idx) for each [Section] block."""
    section = None
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            if section is not None:
                yield section, start, i
            section = s[1:-1]
            start = i + 1
    if section is not None:
        yield section, start, len(lines)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _reflect(v: float, m: float) -> float:
    """Fold a coordinate into [0, m] by reflecting off the boundaries (bounce).

    Unlike clamping, this keeps the amplified jump distance intact and avoids
    piling objects against the walls, which is essential for dense maps.
    """
    if m <= 0:
        return 0.0
    period = 2 * m
    v = v % period
    if v < 0:
        v += period
    return period - v if v > m else v


def amplify_spacing(
    osu_text: str,
    factor: float,
    *,
    playfield: tuple[int, int] = PLAYFIELD,
    max_jump: Optional[float] = None,
) -> str:
    """Scale the movement vector between consecutive objects by ``factor``.

    Bigger jumps raise aim difficulty/star-rating without touching rhythm. The
    new position of each object is anchored to the previous (already moved)
    object plus the scaled original gap, clamped to the playfield. Sliders and
    their control points are translated rigidly by the head delta. Spinners are
    left in place and reset the movement anchor.

    factor == 1.0 returns the input unchanged.
    """
    if factor == 1.0:
        return osu_text

    w, h = playfield
    lines = osu_text.splitlines()
    ho_range = None
    for name, start, end in _iter_sections(lines):
        if name == "HitObjects":
            ho_range = (start, end)
            break
    if ho_range is None:
        return osu_text

    prev_orig: Optional[tuple[float, float]] = None
    prev_new: Optional[tuple[float, float]] = None

    for idx in range(*ho_range):
        line = lines[idx]
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            ox, oy = float(parts[0]), float(parts[1])
            otype = int(parts[3])
        except ValueError:
            continue

        if otype & _SPINNER:
            prev_orig = (ox, oy)
            prev_new = (ox, oy)
            continue

        if prev_orig is None:
            nx, ny = ox, oy
        else:
            dx = (ox - prev_orig[0]) * factor
            dy = (oy - prev_orig[1]) * factor
            if max_jump and max_jump > 0:
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > max_jump:
                    scale = max_jump / dist
                    dx, dy = dx * scale, dy * scale
            nx = _reflect(prev_new[0] + dx, w)
            ny = _reflect(prev_new[1] + dy, h)

        delta = (nx - ox, ny - oy)
        parts[0] = str(int(round(nx)))
        parts[1] = str(int(round(ny)))

        # Slider: translate curve control points in field 5 (e.g. "B|256:192|...").
        if otype & _SLIDER and len(parts) > 5 and "|" in parts[5]:
            segs = parts[5].split("|")
            for j in range(1, len(segs)):
                if ":" in segs[j]:
                    cx, cy = segs[j].split(":")
                    segs[j] = f"{int(round(float(cx) + delta[0]))}:{int(round(float(cy) + delta[1]))}"
            parts[5] = "|".join(segs)

        lines[idx] = ",".join(parts)
        prev_orig = (ox, oy)
        prev_new = (nx, ny)

    trailing = "\n" if osu_text.endswith("\n") else ""
    return "\n".join(lines) + trailing


def amplify_spacing_file(src: str, dst: str, factor: float, **kwargs) -> None:
    with open(src, "r", encoding="utf-8-sig") as f:
        text = f.read()
    out = amplify_spacing(text, factor, **kwargs)
    with open(dst, "w", encoding="utf-8-sig") as f:
        f.write(out)
