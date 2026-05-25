"""Virtual BPM multiplier for a .osu file.

Multiplies the BPM of all uninherited (red) timing points by `factor`
(default 2.0) by dividing their beatLength. Optionally scales the meter so
measure downbeats stay aligned to the original musical bars.

Used as a TIMING-context donor for Mapperatorinator inference:
    python inference.py beatmap_path="'<doubled>.osu'" in_context=[TIMING] ...
The model then sees beat events at `factor`x density and tends to overmap.
"""
import sys
from pathlib import Path


def multiply_bpm(src: str, dst: str, factor: float = 2.0, scale_meter: bool = False) -> int:
    lines = Path(src).read_text(encoding="utf-8").splitlines()
    out, in_tp, changed = [], False, 0
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            in_tp = s == "[TimingPoints]"
            out.append(line)
            continue
        if in_tp and s and not s.startswith("//"):
            parts = s.split(",")
            if len(parts) >= 7 and parts[6].strip() == "1":  # uninherited / red line
                beat = float(parts[1])
                parts[1] = f"{beat / factor:.12g}"
                if scale_meter:
                    parts[2] = str(int(round(int(parts[2]) * factor)))
                out.append(",".join(parts))
                changed += 1
                continue
        out.append(line)
    Path(dst).write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2]
    factor = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    scale_meter = len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "yes")
    n = multiply_bpm(src, dst, factor, scale_meter)
    print(f"Multiplied BPM x{factor} on {n} red timing points -> {dst}")
