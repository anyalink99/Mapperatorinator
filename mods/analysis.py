"""Lightweight audio analysis helpers (no librosa dependency).

Used to decide *where* a song is intense enough to warrant overmapping.
Audio is loaded via pydub (system ffmpeg), energy via numpy.
"""
from __future__ import annotations

import numpy as np
from pydub import AudioSegment


def load_mono(path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 array in [-1, 1] at target_sr."""
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    peak = float(1 << (8 * audio.sample_width - 1))
    return samples / peak, target_sr


def rms_envelope(samples: np.ndarray, sr: int, hop_ms: float = 50.0) -> tuple[np.ndarray, float]:
    """Return (rms_per_frame, hop_seconds) using non-overlapping frames."""
    hop = max(1, int(sr * hop_ms / 1000.0))
    n_frames = len(samples) // hop
    if n_frames == 0:
        return np.array([float(np.sqrt(np.mean(samples ** 2)) if len(samples) else 0.0)]), hop_ms / 1000.0
    trimmed = samples[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    return rms, hop / sr


def energy_at(rms: np.ndarray, hop_s: float, start_ms: float, end_ms: float) -> float:
    """Mean RMS energy over a time window [start_ms, end_ms)."""
    a = int(start_ms / 1000.0 / hop_s)
    b = max(a + 1, int(end_ms / 1000.0 / hop_s))
    a = min(a, len(rms) - 1)
    b = min(b, len(rms))
    if b <= a:
        return float(rms[a]) if 0 <= a < len(rms) else 0.0
    return float(np.mean(rms[a:b]))
