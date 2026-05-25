"""Reusable creative logit processors.

The processor classes live in osuT5 (so the generation hot-path always imports
them) and are re-exported here so any code can build them from a config object.
"""
from __future__ import annotations

from osuT5.osuT5.event import EventType
from osuT5.osuT5.inference.logit_processors import ObjectTypeBias, RhythmDensityBias

__all__ = ["ObjectTypeBias", "RhythmDensityBias", "build_creative_logits_processors"]


def build_creative_logits_processors(tokenizer, *, circle_bias=0.0, slider_bias=0.0,
                                     spinner_bias=0.0, rhythm_density_bias=0.0) -> list:
    """Return the creative logit processors implied by the given biases (empty if all zero)."""
    procs = []
    if circle_bias or slider_bias or spinner_bias:
        procs.append(ObjectTypeBias(tokenizer, {
            EventType.CIRCLE: circle_bias,
            EventType.SLIDER_HEAD: slider_bias,
            EventType.SPINNER: spinner_bias,
        }))
    if rhythm_density_bias:
        procs.append(RhythmDensityBias(tokenizer, rhythm_density_bias))
    return procs
