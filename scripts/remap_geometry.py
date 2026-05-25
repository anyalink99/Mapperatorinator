"""CLI: amplify jump spacing on a finished .osu (raises aim difficulty/star-rating).

    python scripts/remap_geometry.py in.osu out.osu --factor 1.5 [--max-jump 380]

Scales the movement vector between consecutive objects; rhythm is untouched.
osu!standard only.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mods import geometry  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--max-jump", type=float, default=0.0, help="Cap amplified jump distance in px (0 = off)")
    args = ap.parse_args()

    geometry.amplify_spacing_file(
        args.src, args.dst, args.factor, max_jump=(args.max_jump or None)
    )
    print(f"Amplified spacing x{args.factor} -> {args.dst}")


if __name__ == "__main__":
    main()
