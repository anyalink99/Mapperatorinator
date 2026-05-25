"""CLI: inject sub-beat clicks into a song so the model maps extra onsets.

    python scripts/inject_onsets.py song.mp3 donor.osu out.wav --division 2

`donor.osu` provides the beat grid (any .osu with timing; a variable_bpm donor
works too). division=2 adds a click halfway between beats, 4 adds the off-beats.
Feed the resulting audio to inference as audio_path.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mods import audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("timing_osu")
    ap.add_argument("out")
    ap.add_argument("--division", type=int, default=2)
    ap.add_argument("--include-beats", action="store_true", help="Also click on the beats")
    ap.add_argument("--only-beats", action="store_true", help="Metronome: click only beats (verify grid)")
    ap.add_argument("--dense", action="store_true", help="Click only in energetic sections of the song")
    ap.add_argument("--percentile", type=float, default=60.0, help="Energy percentile for --dense (loudest 100-P%% get clicks)")
    ap.add_argument("--segment-beats", type=int, default=4, help="Section length in beats for --dense")
    ap.add_argument("--gain-db", type=float, default=-8.0)
    ap.add_argument("--freq", type=float, default=1200.0)
    ap.add_argument("--dur-ms", type=float, default=25.0)
    ap.add_argument("--kind", choices=["sine", "noise"], default="sine")
    args = ap.parse_args()

    if args.dense:
        out = audio.inject_dense_file(
            args.audio, args.timing_osu, args.out, division=args.division,
            energy_percentile=args.percentile, segment_beats=args.segment_beats,
            include_beats=args.include_beats,
            gain_db=args.gain_db, click_freq=args.freq, click_dur_ms=args.dur_ms, click_kind=args.kind,
        )
        print(f"Injected dense 1/{args.division} clicks (top {100 - args.percentile:.0f}% energy) -> {out}")
    else:
        out = audio.inject_subdivisions_file(
            args.audio, args.timing_osu, args.out, division=args.division,
            include_beats=args.include_beats, only_beats=args.only_beats,
            gain_db=args.gain_db, click_freq=args.freq, click_dur_ms=args.dur_ms, click_kind=args.kind,
        )
        mode = "beats only" if args.only_beats else (f"1/{args.division}+beats" if args.include_beats else f"1/{args.division}")
        print(f"Injected {mode} clicks -> {out}")


if __name__ == "__main__":
    main()
