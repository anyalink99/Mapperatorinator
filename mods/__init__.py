"""mods: reusable creative-control tools for Mapperatorinator.

Four levers for steering generation beyond CLI settings, all without retraining:

- logits   : object-type & rhythm-density logit processors (circles vs sliders, density)
- geometry : spacing/jump amplifier on a finished .osu (raises aim difficulty)
- timing   : virtual-BPM donors (uniform or energy-aware "section-aware" doubling)
- audio    : onset/click injection so the model hears (and maps) extra sub-beats
- analysis : audio energy helpers shared by the above

Each module exposes pure functions plus *_file helpers, and is wired into the
inference pipeline via config flags (see configs/inference/default.yaml) or usable
standalone via scripts/.
"""
from . import analysis, audio, geometry, timing  # noqa: F401

__all__ = ["analysis", "audio", "geometry", "timing", "logits"]
