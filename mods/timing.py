"""Virtual-BPM timing mods: build a TIMING-context donor .osu whose beat grid is
denser than the song's real BPM, so the model overmaps. Two strategies:

- uniform_multiply: multiply BPM everywhere by a constant factor.
- variable_bpm_donor: multiply only where the song is energetic ("where needed"),
  keeping calm sections at the real BPM for a natural dense/calm contrast.

Feed the result to inference with ``beatmap_path='<donor>.osu' in_context=[TIMING]``.
"""
from __future__ import annotations

from typing import Optional

from . import analysis


def _split_sections(text: str) -> list[str]:
    return text.splitlines()


def _find_section(lines: list[str], name: str) -> Optional[tuple[int, int]]:
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == f"[{name}]":
            start = i + 1
        elif start is not None and s.startswith("[") and s.endswith("]"):
            return start, i
    if start is not None:
        return start, len(lines)
    return None


def _parse_red_points(lines: list[str]) -> list[list[str]]:
    """Return uninherited (red) timing-point lines split into fields, time-sorted."""
    sec = _find_section(lines, "TimingPoints")
    reds = []
    if sec:
        for idx in range(*sec):
            s = lines[idx].strip()
            if not s or s.startswith("//"):
                continue
            p = s.split(",")
            if len(p) >= 7 and p[6].strip() == "1":
                reds.append(p)
    reds.sort(key=lambda p: float(p[0]))
    return reds


def _last_object_time(lines: list[str]) -> Optional[float]:
    sec = _find_section(lines, "HitObjects")
    if not sec:
        return None
    last = None
    for idx in range(*sec):
        s = lines[idx].strip()
        if s and s[0].isdigit():
            last = float(s.split(",")[2])
    return last


def _replace_timing_section(lines: list[str], new_red_lines: list[str]) -> str:
    sec = _find_section(lines, "TimingPoints")
    if not sec:
        return "\n".join(lines) + "\n"
    out = lines[: sec[0]] + new_red_lines + lines[sec[1]:]
    return "\n".join(out) + "\n"


def uniform_multiply(osu_text: str, factor: float = 2.0, scale_meter: bool = False) -> str:
    """Multiply BPM of every red timing point by ``factor`` (beatLength / factor)."""
    lines = _split_sections(osu_text)
    sec = _find_section(lines, "TimingPoints")
    if not sec:
        return osu_text
    for idx in range(*sec):
        s = lines[idx].strip()
        if not s or s.startswith("//"):
            continue
        p = s.split(",")
        if len(p) >= 7 and p[6].strip() == "1":
            p[1] = f"{float(p[1]) / factor:.12g}"
            if scale_meter:
                p[2] = str(int(round(int(p[2]) * factor)))
            lines[idx] = ",".join(p)
    trailing = "\n" if osu_text.endswith("\n") else ""
    return "\n".join(lines) + trailing


def _beat_grid(reds: list[list[str]], song_end: float) -> list[tuple[float, float, int, list[str]]]:
    """Expand red points into per-beat (time, beatLength, meter, template) tuples."""
    beats = []
    for i, p in enumerate(reds):
        offset = float(p[0])
        beat_len = float(p[1])
        meter = int(p[2])
        region_end = float(reds[i + 1][0]) if i + 1 < len(reds) else song_end
        t = offset
        # guard against pathological tiny beat lengths
        if beat_len <= 0:
            continue
        while t < region_end - 1e-6:
            beats.append((t, beat_len, meter, p))
            t += beat_len
    return beats


def variable_bpm_donor(
    osu_text: str,
    audio_path: str,
    *,
    high_factor: float = 2.0,
    low_factor: float = 1.0,
    segment_beats: int = 4,
    energy_percentile: float = 60.0,
    scale_meter: bool = False,
) -> str:
    """Build a donor that doubles (high_factor) energetic segments and leaves calm
    ones at low_factor. Segment boundaries land on real downbeats so the grid
    stays aligned. Returns modified .osu text (red points only).
    """
    lines = _split_sections(osu_text)
    reds = _parse_red_points(lines)
    if not reds:
        return osu_text

    song_end = _last_object_time(lines)
    if song_end is None:
        song_end = float(reds[-1][0]) + 8 * float(reds[-1][1])

    beats = _beat_grid(reds, song_end)
    if not beats:
        return osu_text

    import numpy as np
    rms, hop_s = analysis.rms_envelope(*analysis.load_mono(audio_path))

    # Chunk beats into segments WITHOUT crossing original BPM-change boundaries, so
    # a segment's beatLength/meter always belongs to a single source timing region.
    segments = []  # (start_time, beat_len, meter, template, energy)
    i = 0
    while i < len(beats):
        region = beats[i][3]
        chunk = []
        while i < len(beats) and beats[i][3] is region and len(chunk) < segment_beats:
            chunk.append(beats[i])
            i += 1
        start_t = chunk[0][0]
        end_t = chunk[-1][0] + chunk[-1][1]
        energy = analysis.energy_at(rms, hop_s, start_t, end_t)
        segments.append((start_t, chunk[0][1], chunk[0][2], region, energy))

    threshold = float(np.percentile([s[4] for s in segments], energy_percentile))

    new_red_lines: list[str] = []
    last_len = None
    last_region = None
    for start_t, beat_len, meter, region, energy in segments:
        factor = high_factor if energy >= threshold else low_factor
        target_len = beat_len / factor
        target_meter = int(round(meter * factor)) if scale_meter else meter
        # Emit a red point when the tempo changes OR a new source region begins
        # (re-anchors the grid even if the post-factor beatLength coincides).
        if last_region is region and last_len is not None and abs(target_len - last_len) < 1e-6:
            continue
        p = list(region)
        p[0] = str(int(round(start_t)))
        p[1] = f"{target_len:.12g}"
        p[2] = str(target_meter)
        p[6] = "1"
        if len(p) >= 8:
            p[7] = "0"
        new_red_lines.append(",".join(p))
        last_len = target_len
        last_region = region

    return _replace_timing_section(lines, new_red_lines)


# ---- file helpers ----

def uniform_multiply_file(src: str, dst: str, factor: float = 2.0, scale_meter: bool = False) -> None:
    with open(src, "r", encoding="utf-8-sig") as f:
        text = f.read()
    with open(dst, "w", encoding="utf-8-sig") as f:
        f.write(uniform_multiply(text, factor, scale_meter))


def variable_bpm_donor_file(src: str, dst: str, audio_path: str, **kwargs) -> None:
    with open(src, "r", encoding="utf-8-sig") as f:
        text = f.read()
    with open(dst, "w", encoding="utf-8-sig") as f:
        f.write(variable_bpm_donor(text, audio_path, **kwargs))
