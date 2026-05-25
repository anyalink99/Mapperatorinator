"""Print star rating and basic stats for one or more .osu files (rosu-pp)."""
import sys
from pathlib import Path
import rosu_pp_py as rosu


def stats(path: str):
    text = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    objs = []
    in_ho = False
    for line in text:
        s = line.strip()
        if s.startswith("["):
            in_ho = s == "[HitObjects]"
            continue
        if in_ho and s and s[0].isdigit():
            objs.append(s)
    circles = sliders = spinners = 0
    for o in objs:
        t = int(o.split(",")[3])
        if t & 1:
            circles += 1
        elif t & 2:
            sliders += 1
        elif t & 8:
            spinners += 1
    last_ms = float(objs[-1].split(",")[2]) if objs else 0
    m = rosu.Beatmap(path=path)
    stars = rosu.Difficulty().calculate(m).stars
    dur_s = last_ms / 1000
    dens = len(objs) / dur_s if dur_s else 0
    print(f"{Path(path).name}")
    print(f"  stars : {stars:.2f}")
    print(f"  objects: {len(objs)} (circles {circles}, sliders {sliders}, spinners {spinners})")
    print(f"  span   : {dur_s:.1f}s   density: {dens:.2f} obj/s")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
