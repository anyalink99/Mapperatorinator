"""Onset injection: add short percussive clicks to a song so the model *hears*
extra transients on chosen subdivisions and overmaps them musically (a deeper
alternative to faking the timing grid).

Produces an augmented audio file that is fed to inference normally.
"""
from __future__ import annotations

import numpy as np
from pydub import AudioSegment

from . import analysis
from . import timing as _timing


def synth_click(sr: int, dur_ms: float = 25.0, freq: float = 1200.0, kind: str = "sine") -> np.ndarray:
    """Synthesize a short exponentially-decaying click in [-1, 1]."""
    n = max(1, int(sr * dur_ms / 1000.0))
    t = np.arange(n) / sr
    env = np.exp(-t * (4000.0 / max(1.0, dur_ms)))
    if kind == "noise":
        wave = np.random.uniform(-1, 1, n)
    else:
        wave = np.sin(2 * np.pi * freq * t)
    click = (wave * env).astype(np.float32)
    peak = float(np.max(np.abs(click))) or 1.0
    return click / peak


def grid_times(
    osu_text: str,
    division: int = 2,
    *,
    include_beats: bool = False,
    only_beats: bool = False,
    song_end: float | None = None,
) -> list[float]:
    """Times (ms) on the beat grid of ``osu_text``.

    - only_beats=True: just the beats themselves (metronome; use to verify the
      grid actually lines up with the song).
    - otherwise: the sub-beat onsets between beats (division=2 -> the "and"
      eighth; division=4 -> the 1/4 & 3/4), optionally plus the beats themselves
      when include_beats=True.

    IMPORTANT: pass a *real-BPM* .osu here, not a doubled-BPM donor, or the
    clicks land on syncopated positions and sound off-beat.
    """
    lines = osu_text.splitlines()
    reds = _timing._parse_red_points(lines)
    if not reds:
        return []
    if song_end is None:
        song_end = _timing._last_object_time(lines) or (float(reds[-1][0]) + 8 * float(reds[-1][1]))
    times: list[float] = []
    for time, beat_len, _meter, _tmpl in _timing._beat_grid(reds, song_end):
        if only_beats or include_beats:
            times.append(time)
        if not only_beats:
            step = beat_len / division
            for k in range(1, division):
                times.append(time + k * step)
    return sorted(times)


def subdivision_times(osu_text: str, division: int = 2, song_end: float | None = None) -> list[float]:
    """Backwards-compatible alias: sub-beat onsets only."""
    return grid_times(osu_text, division, song_end=song_end)


def dense_grid_times(
    osu_text: str,
    audio_path: str,
    division: int = 2,
    *,
    energy_percentile: float = 60.0,
    segment_beats: int = 4,
    include_beats: bool = False,
    song_end: float | None = None,
) -> list[float]:
    """Sub-beat onset times ONLY inside the song's energetic ("dense") sections.

    This is the honest, audio-side version of section-aware overmapping: instead
    of faking a denser BPM grid in loud parts, we add real clicks there so the
    model overmaps the dense moments and leaves calm parts alone. Segments never
    cross a BPM-change boundary. energy_percentile controls how much of the song
    counts as "dense" (60 -> loudest ~40% of segments get clicks).
    """
    lines = osu_text.splitlines()
    reds = _timing._parse_red_points(lines)
    if not reds:
        return []
    if song_end is None:
        song_end = _timing._last_object_time(lines) or (float(reds[-1][0]) + 8 * float(reds[-1][1]))
    beats = _timing._beat_grid(reds, song_end)
    if not beats:
        return []

    import numpy as np
    rms, hop_s = analysis.rms_envelope(*analysis.load_mono(audio_path))

    segments = []  # (chunk, energy)
    i = 0
    while i < len(beats):
        region = beats[i][3]
        chunk = []
        while i < len(beats) and beats[i][3] is region and len(chunk) < segment_beats:
            chunk.append(beats[i])
            i += 1
        st, en = chunk[0][0], chunk[-1][0] + chunk[-1][1]
        segments.append((chunk, analysis.energy_at(rms, hop_s, st, en)))

    threshold = float(np.percentile([e for _, e in segments], energy_percentile))

    times: list[float] = []
    for chunk, energy in segments:
        if energy < threshold:
            continue
        for time, beat_len, _meter, _tmpl in chunk:
            if include_beats:
                times.append(time)
            step = beat_len / division
            for k in range(1, division):
                times.append(time + k * step)
    return sorted(times)


def inject_dense_file(
    audio_path: str,
    timing_osu: str,
    out_path: str,
    division: int = 2,
    *,
    energy_percentile: float = 60.0,
    segment_beats: int = 4,
    include_beats: bool = False,
    **click_kwargs,
) -> str:
    """Click the sub-beats of a (real-BPM) .osu into the audio, but only inside
    energetic sections (uses the same audio for energy analysis)."""
    with open(timing_osu, "r", encoding="utf-8-sig") as f:
        osu_text = f.read()
    times = dense_grid_times(
        osu_text, audio_path, division,
        energy_percentile=energy_percentile, segment_beats=segment_beats, include_beats=include_beats,
    )
    return inject_clicks(audio_path, out_path, times, **click_kwargs)


def inject_clicks(
    audio_path: str,
    out_path: str,
    times_ms: list[float],
    *,
    gain_db: float = -8.0,
    click_dur_ms: float = 25.0,
    click_freq: float = 1200.0,
    click_kind: str = "sine",
) -> str:
    """Mix clicks into the audio at the given times, preserving sample rate and
    channels. Output format follows the out_path extension (.wav recommended;
    .mp3/.ogg need ffmpeg)."""
    seg = AudioSegment.from_file(audio_path)
    sr, ch, sw = seg.frame_rate, seg.channels, seg.sample_width
    max_int = float(1 << (8 * sw - 1))

    samples = np.array(seg.get_array_of_samples()).astype(np.float32).reshape(-1, ch)
    click = synth_click(sr, click_dur_ms, click_freq, click_kind) * (10 ** (gain_db / 20.0)) * max_int

    clen = len(click)
    n_frames = samples.shape[0]
    click_col = click[:, None]
    for t in times_ms:
        i = int(t / 1000.0 * sr)
        if i < 0 or i >= n_frames:
            continue
        j = min(i + clen, n_frames)
        samples[i:j, :] += click_col[: j - i]

    np.clip(samples, -max_int, max_int - 1, out=samples)
    out_seg = AudioSegment(
        samples.astype(_int_dtype(sw)).tobytes(), frame_rate=sr, sample_width=sw, channels=ch
    )
    fmt = out_path.rsplit(".", 1)[-1].lower()
    out_seg.export(out_path, format="wav" if fmt not in ("mp3", "ogg", "flac", "wav") else fmt)
    return out_path


def _int_dtype(sample_width: int):
    return {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)


def inject_subdivisions_file(
    audio_path: str,
    timing_osu: str,
    out_path: str,
    division: int = 2,
    *,
    include_beats: bool = False,
    only_beats: bool = False,
    **click_kwargs,
) -> str:
    """Convenience: read a (real-BPM) .osu, click its grid into the audio."""
    with open(timing_osu, "r", encoding="utf-8-sig") as f:
        osu_text = f.read()
    times = grid_times(osu_text, division, include_beats=include_beats, only_beats=only_beats)
    return inject_clicks(audio_path, out_path, times, **click_kwargs)
