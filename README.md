# Mapperatorinator — Creative Control Fork

A fork of [OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) that adds a
small, reusable **creative-control toolkit** (the [`mods/`](mods/) package) for steering beatmap
generation *beyond what the base model produces on its own* — without retraining.

> Upstream Mapperatorinator generates rankable osu! beatmaps from raw audio. It is excellent, but
> like any generative model it stays close to its training distribution. For a calm or slow song it
> will keep producing ~4★ maps no matter how high you set `difficulty`, and it leans heavily on
> sliders. This fork is about **pushing past that ceiling on purpose**: overmapping, higher star
> rating, and a jump-heavy aim style — as controllable knobs.

## The goal

Take a slow/normal song and turn it into a **high star-rating, jump-heavy** map (think "double-BPM
overmap"), the way a human mapper would — and do it through reusable, composable tools rather than
one-off hacks. The long-term goal is genuine 9–10★ generation; the short-term tools below already
move a 125 BPM song from the model's hard ceiling of ~3.8★ up to ~7.5★ playable.

## What this fork adds

All levers are off by default (the fork is a drop-in superset of upstream) and can be combined.

| Lever | What it does | Where |
|---|---|---|
| **Virtual BPM multiplier** | Builds a TIMING-context donor whose beat grid is 2–4× denser than the song's real BPM, so the model overmaps to the faster grid. Uniform or energy-aware ("section-aware") variants. | [`mods/timing.py`](mods/timing.py), [`scripts/variable_bpm.py`](scripts/variable_bpm.py) |
| **Onset / click injection** | Mixes real percussive clicks into the audio on chosen sub-beats so the model *hears* (and maps) extra onsets — honest overmapping. Supports **dense-only mode**: clicks added only in energetic sections. | [`mods/audio.py`](mods/audio.py), [`scripts/inject_onsets.py`](scripts/inject_onsets.py) |
| **Object-type steering** | Logit bias that shifts generation toward circles / away from sliders (or vice-versa), turning slider-heavy "tech" output into jump/stream-heavy maps. Gated to type-decision steps so it never corrupts timing. | `circle_bias` / `slider_bias` / `spinner_bias`, [`logit_processors.py`](osuT5/osuT5/inference/logit_processors.py) |
| **Rhythm-density bias** | Ramp bias on time-shift tokens favouring smaller deltas → denser rhythms. | `rhythm_density_bias` |
| **Jump-spacing amplifier** | Post-process that scales the movement vector between objects (bigger jumps → higher aim difficulty), **reflecting** off the playfield edges instead of clamping, so dense maps don't pile in the corners. | `spacing_multiplier`, [`mods/geometry.py`](mods/geometry.py), [`scripts/remap_geometry.py`](scripts/remap_geometry.py) |
| **Clean-audio export** | `osz_audio_path` lets the model hear a tool track (e.g. click-injected audio) while the exported `.osz` bundles the clean original. | `osz_audio_path` |

Each module is usable three ways: as a **config flag** to `inference.py`, as a **standalone CLI**
in [`scripts/`](scripts/), and as a **library** (`from mods import timing, audio, geometry, logits`).

## Quick start

Install per [upstream instructions](https://github.com/OliBomby/Mapperatorinator#installation)
(Python 3.10, ffmpeg, a CUDA build of PyTorch — note: match the CUDA build to your driver, e.g.
`cu128` for driver ≤ 12.9), then:

```sh
pip install -r requirements.txt
```

### Example: overmap a slow song into a jump-heavy map

```sh
# 1. Double the song's BPM into a timing donor (use the song's real timing as source)
python scripts/variable_bpm.py reference.osu donor_x2.osu --factor 2

# 2. (optional) inject 1/4 clicks into the audio, only in the loud sections,
#    so the model hears stream-density where the song is intense
python scripts/inject_onsets.py song.mp3 reference.osu song_clicks.wav \
    --dense --division 4 --include-beats --percentile 60

# 3. Generate, steering toward circles + amplifying jumps, bundling the clean mp3
python inference.py \
    audio_path="'song_clicks.wav'" osz_audio_path="'song.mp3'" \
    beatmap_path="'donor_x2.osu'" in_context=[TIMING] output_type=[MAP,SV] \
    gamemode=0 difficulty=12 slider_multiplier=2.2 \
    slider_bias=-22 circle_bias=14 spacing_multiplier=1.6 \
    export_osz=true
```

Measure the result:

```sh
python scripts/sr.py output/beatmap*.osu   # star rating + object/density stats (rosu-pp)
```

### New config knobs

```yaml
# configs/inference/default.yaml
circle_bias: 0.0            # logit bias toward circles (use large values ~10-30; model is confident)
slider_bias: 0.0            # logit bias away from sliders (e.g. -30)
spinner_bias: 0.0
rhythm_density_bias: 0.0    # ramp bias toward denser rhythm
spacing_multiplier: 1.0     # jump-spacing amplifier (reflection-based), 1.0 = off
spacing_max_jump: 0         # optional cap (px) on amplified jumps
osz_audio_path: ''          # clean audio for the .osz when audio_path is a tool track
```

## Findings

Measured with `scripts/sr.py` (rosu-pp). Test track: a 125 BPM song the base model caps at ~3.8★.

| Configuration | Stars | Circles / sliders |
|---|---|---|
| Vanilla, `difficulty=5` | 3.82 | 49% circles |
| BPM ×2 + `difficulty=9` | 5.72 | 38% |
| BPM ×2 + `difficulty=12` + high SV | 6.27 | 37% |
| + jump-spacing amplifier (reflection, ×1.6) | 7.25 | — |
| Section-aware ×2 + jump bias + spacing (showcase) | 7.45 | **56%** circles |

Takeaways:
- **Virtual BPM doubling is the strongest single lever** (+~1.9★) — it makes the model overmap.
- **SV and jump-spacing** stack on top for aim difficulty.
- **Object-type steering** flips the *character* from slider-tech to jump/stream (the model is very
  confident about types, so biases need magnitude ~15–30 and must be gated to type steps).
- **Descriptors change style, not intensity** — they did not raise SR on their own.

## Limits & roadmap

Parameters get us to ~7–7.5★ but **not to a clean 9–10★**: that lies outside the base model's
training distribution (few very-high-SR maps exist in the data, and slow BPM correlates with low SR).
The next step is **LoRA fine-tuning** on a high-SR / jump-heavy dataset
(see [`configs/train/lora_v32.yaml`](configs/train/lora_v32.yaml)) to actually shift the distribution,
with these mods layered on top at inference time.

## Credits

This fork stands entirely on [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) by
OliBomby, built on [osuT5](https://github.com/gyataro/osuT5) and
[osu-diffusion](https://github.com/OliBomby/osu-diffusion). All credit for the model, training code,
and base pipeline goes to them. **Use AI mapping responsibly and always disclose it.**
