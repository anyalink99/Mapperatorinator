"""CLI: build a virtual-BPM TIMING donor from a reference .osu.

Uniform doubling:
    python scripts/variable_bpm.py in.osu donor.osu --factor 2

Section-aware (energy-driven) doubling, needs the song:
    python scripts/variable_bpm.py in.osu donor.osu --audio song.mp3 \
        --high 2 --low 1 --segment-beats 4 --percentile 60

Then feed the donor to inference:
    python inference.py beatmap_path="'donor.osu'" in_context=[TIMING] ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mods import timing  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--audio", help="Song path; enables section-aware mode")
    ap.add_argument("--factor", type=float, default=2.0, help="Uniform BPM factor (no --audio)")
    ap.add_argument("--high", type=float, default=2.0, help="Factor for energetic segments")
    ap.add_argument("--low", type=float, default=1.0, help="Factor for calm segments")
    ap.add_argument("--segment-beats", type=int, default=4)
    ap.add_argument("--percentile", type=float, default=60.0, help="Energy percentile -> high factor above it")
    ap.add_argument("--scale-meter", action="store_true")
    args = ap.parse_args()

    if args.audio:
        timing.variable_bpm_donor_file(
            args.src, args.dst, args.audio,
            high_factor=args.high, low_factor=args.low,
            segment_beats=args.segment_beats, energy_percentile=args.percentile,
            scale_meter=args.scale_meter,
        )
        print(f"Section-aware donor (high x{args.high}/low x{args.low}) -> {args.dst}")
    else:
        timing.uniform_multiply_file(args.src, args.dst, args.factor, args.scale_meter)
        print(f"Uniform BPM x{args.factor} -> {args.dst}")


if __name__ == "__main__":
    main()
