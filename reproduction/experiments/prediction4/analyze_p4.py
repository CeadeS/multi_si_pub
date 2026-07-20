#!/usr/bin/env python3
"""Analyze Prediction-4 one-shot mediated-game episodes."""
import json, math, sys
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

def main(path="episodes.json", out="RESULTS_P4.md"):
    eps = json.load(open(path))
    eps = [e for e in eps if e.get("action") in ("COOPERATE", "DEFECT")]
    cells = defaultdict(lambda: [0, 0])
    for e in eps:
        k = (e["model"], e["d"], e["mediator"])
        cells[k][1] += 1
        if e["action"] == "COOPERATE": cells[k][0] += 1
    lines = ["# Prediction-4 results", "",
             f"Valid episodes: {len(eps)}", "",
             "| model | D_pts | mediator | n | compliance | Wilson 95% CI |",
             "|---|---:|---|---:|---:|---|"]
    for (m, d, med) in sorted(cells):
        c, n = cells[(m, d, med)]
        lo, hi = wilson(c, n)
        lines.append(f"| {m} | {d} | {'yes' if med else 'no'} | {n} | {c/n:.2f} | [{lo:.2f}, {hi:.2f}] |")
    # headline aggregates
    def agg(pred):
        c = sum(v[0] for k, v in cells.items() if pred(k)); n = sum(v[1] for k, v in cells.items() if pred(k))
        return c, n
    for name, pred in [
        ("patience-free (D<60), mediated", lambda k: k[1] < 60 and k[2]),
        ("patience-free (D<60), unmediated", lambda k: k[1] < 60 and not k[2]),
        ("boundary (D=60), mediated", lambda k: k[1] == 60 and k[2]),
        ("boundary (D=60), unmediated", lambda k: k[1] == 60 and not k[2]),
        ("dilemma (D>60), mediated", lambda k: k[1] > 60 and k[2]),
        ("dilemma (D>60), unmediated", lambda k: k[1] > 60 and not k[2]),
    ]:
        c, n = agg(pred)
        lo, hi = wilson(c, n)
        lines.append("")
        lines.append(f"**{name}**: {c}/{n} = {c/n:.3f} [{lo:.3f}, {hi:.3f}]")
    open(out, "w").write("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main(*sys.argv[1:])
