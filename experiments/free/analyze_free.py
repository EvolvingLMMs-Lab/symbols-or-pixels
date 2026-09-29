#!/usr/bin/env python3
"""Summarize the free-reasoning pilot (experiments/free/free_pilot.py) on the 475 paired samples.

Per model, at the 2,048-token view (`*_cap2048`, the note's budget) and the 8,192-token run:
    greedy    solve rate and outcomes per prompt (layout, direct, free), median completion tokens, the share of
              outputs with reasoning before the ANSWER line, and the paired differences between prompts
    local     the released vLLM runs (results/per_sample.csv) for layout and direct against the API runs
    sampled   per prompt with k sampled draws: mean per-draw rate, range over draws, mean draw - greedy, any-of-k
              (an oracle upper bound), the share of samples whose draws disagree on correctness; the paired
              mean-draw differences between prompts; an exact-answer plurality vote on non-pixel tasks against
              the mean of the same draws; and, for 27B, paired text - G27 differences at v2 >= 0.9 and >= 0.7
Conventions follow analysis/analyze.py: task-weighted rates, 4,000 task-bootstrap resamples with random.Random(0)
(fresh per statistic), nearest-rank percentiles. Outcomes follow analysis/build_results.py.

Usage:
    python3 experiments/free/analyze_free.py [--models qwen3.6-27b,qwen3.5-9b]
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/free_pilot"
B, SEED = 4000, 0
HINTS = ("layout", "direct", "free")
PAIRS = (("free", "layout"), ("free", "direct"), ("direct", "layout"))
CAPS = (("2,048", "_cap2048"), ("8,192", ""))
VIDEO = {"qwen3.6-27b": "g27"}  # the note's size-matched pair; Qwen3.5-9B has none


def outcome(g):
    if g["correct"]:
        return "correct"
    no_answer = str(g.get("why", "")).startswith("no answer")
    if no_answer and g.get("finish") == "length":
        return "truncated"
    return "no_answer" if no_answer else "wrong"


def load(name, n):
    path = RUNS / name / "graded.jsonl"
    if not path.exists():
        return None
    run = {(g["task"], g["idx"]): dict(g, outcome=outcome(g)) for g in map(json.loads, path.read_text().splitlines())}
    return run if len(run) >= n else None


class Paired:
    def __init__(self, rows):
        tasks = defaultdict(list)
        for k, r in enumerate(rows):
            tasks[r["task"]].append(k)
        self.tasks = list(tasks.values())
        rng = random.Random(SEED)
        n = len(self.tasks)
        self.boot = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(B)])

    def per_task(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([x[t].mean() for t in self.tasks])

    def rate(self, x):
        t = self.per_task(x)
        lo, hi = np.percentile(t[self.boot].mean(axis=1), [2.5, 97.5], method="nearest")
        return t.mean(), lo, hi


def fmt(r, pp=False):
    m, lo, hi = r
    if pp:
        return f"{100 * m:+5.1f} pp [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
    return f"{100 * m:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}]"


def col(run, rows, key="correct"):
    return np.array([run[(r["task"], int(r["idx"]))][key] for r in rows])


def reasoned_share(name, rows):
    recs = {(r["task"], r["idx"]): r for r in map(json.loads, (RUNS / name / "responses.jsonl").read_text().splitlines())
            if "error" not in r}
    pre = []
    for r in rows:
        c = recs[(r["task"], int(r["idx"]))]["content"]
        pre.append(len(c.rsplit("ANSWER:", 1)[0].strip()) if "ANSWER:" in c else len(c))
    return np.mean([n > 100 for n in pre])


def vote(runs, rows, ks):
    """Exact-answer plurality over the parsed answers of each sample's draws; ties go to the earliest draw."""
    out, distinct = [], Counter()
    for k in ks:
        r = rows[k]
        answers = [(json.dumps(g["answer"], sort_keys=True), bool(g["correct"]))
                   for g in (run[(r["task"], int(r["idx"]))] for run in runs) if g["answer"] is not None]
        distinct[len({a for a, _ in answers})] += 1
        if not answers:
            out.append(False)
            continue
        counts = Counter(a for a, _ in answers)
        winner = next(a for a, _ in answers if counts[a] == max(counts.values()))
        out.append(next(ok for a, ok in answers if a == winner))
    return np.array(out, float), distinct


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="qwen3.6-27b,qwen3.5-9b")
    args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / "results/per_sample.csv").open()))
    n = len(rows)
    P = Paired(rows)
    nonpx = [k for k, r in enumerate(rows) if r["answer_kind"] == "non_pixel"]
    for m in args.models.split(","):
        print(f"\n######## {m}")
        for cap_name, suffix in CAPS:
            greedy = {h: load(f"{m}_{h}_greedy{suffix}", n) for h in HINTS}
            if any(v is None for v in greedy.values()):
                print(f"  greedy @ {cap_name}: incomplete")
                continue
            print(f"  --- greedy, cap {cap_name}")
            c = {h: col(greedy[h], rows).astype(float) for h in HINTS}
            for h in HINTS:
                oc = Counter(col(greedy[h], rows, "outcome"))
                toks = col(greedy[h], rows, "completion_tokens").astype(float)
                extra = f"  reasoning before ANSWER {100 * reasoned_share(f'{m}_{h}_greedy', rows):.1f}%" if not suffix else ""
                print(f"    {h:7s} {fmt(P.rate(c[h]))}  correct {oc['correct']:3d} wrong {oc['wrong']:3d} "
                      f"no_answer {oc['no_answer']:3d} truncated {oc['truncated']:3d}  median tokens {np.median(toks):.0f}{extra}")
            for a, b in PAIRS:
                cells = Counter(zip(c[a], c[b]))
                print(f"    {a} - {b}: {fmt(P.rate(c[a] - c[b]), pp=True)}   only {a} {cells[(1.0, 0.0)]}, "
                      f"only {b} {cells[(0.0, 1.0)]}")
            if suffix:
                for h in ("layout", "direct"):
                    local = np.array([r[f"{m}_{h}_correct"] == "1" for r in rows], float)
                    print(f"    local vLLM {h}: {fmt(P.rate(local))}; API - local {fmt(P.rate(c[h] - local), pp=True)}; "
                          f"same outcome on {100 * np.mean(local == c[h]):.1f}% of samples")
        for cap_name, suffix in CAPS:
            mean_draw, have = {}, []
            for h in HINTS:
                draws = sorted(p.name.split("_")[-1] for p in RUNS.glob(f"{m}_{h}_s*") if not p.name.endswith("_cap2048"))
                runs = [load(f"{m}_{h}_{d}{suffix}", n) for d in draws]
                greedy = load(f"{m}_{h}_greedy{suffix}", n)
                if not draws or greedy is None or any(r is None for r in runs):
                    print(f"  sampled {h} @ {cap_name}: incomplete ({len(draws)} draw directories)")
                    continue
                have.append(h)
                k = len(runs)
                C = np.array([col(r, rows) for r in runs], float)
                g = col(greedy, rows).astype(float)
                mean_draw[h] = C.mean(axis=0)
                per_draw = [P.per_task(C[i]).mean() for i in range(k)]
                s = C.sum(axis=0)
                print(f"  --- sampled {h}, {k} draws (T 0.7, top_p 0.8, top_k 20), cap {cap_name}")
                print(f"    greedy {fmt(P.rate(g))} | mean draw {fmt(P.rate(mean_draw[h]))}, draws "
                      f"{100 * min(per_draw):.1f}-{100 * max(per_draw):.1f}% | mean draw - greedy {fmt(P.rate(mean_draw[h] - g), pp=True)}")
                print(f"    any of {k} (oracle) {fmt(P.rate(C.max(axis=0)))} | all {k} {fmt(P.rate(C.min(axis=0)))} | "
                      f"draws disagree on {100 * np.mean((s > 0) & (s < k)):.1f}% of samples")
                v, distinct = vote(runs, rows, nonpx)
                Pn = Paired([rows[kk] for kk in nonpx])
                gn = g[nonpx]
                print(f"    non-pixel ({len(Pn.tasks)} tasks): greedy {fmt(Pn.rate(gn))}, mean draw "
                      f"{fmt(Pn.rate(mean_draw[h][nonpx]))}, vote {fmt(Pn.rate(v))}")
                print(f"      vote - mean draw {fmt(Pn.rate(v - mean_draw[h][nonpx]), pp=True)}, vote - greedy "
                      f"{fmt(Pn.rate(v - gn), pp=True)}; distinct parsed answers per sample {dict(sorted(distinct.items()))}")
            if len(have) > 1:
                print(f"  --- sampled, paired mean-draw differences, cap {cap_name}")
                for a, b in PAIRS:
                    if a in mean_draw and b in mean_draw:
                        print(f"    {a} - {b}: {fmt(P.rate(mean_draw[a] - mean_draw[b]), pp=True)}")
            if VIDEO.get(m) and len(have) == len(HINTS):
                print(f"  --- text - {VIDEO[m]} (paired, mixed stack: API text, local video), cap {cap_name}")
                v2 = np.array([float(r[f"{VIDEO[m]}_v2"]) for r in rows])
                for cut in (0.9, 0.7):
                    vid = (v2 >= cut).astype(float)
                    parts = []
                    for h in HINTS:
                        g = col(load(f"{m}_{h}_greedy{suffix}", n), rows).astype(float)
                        parts.append(f"{h} greedy {fmt(P.rate(g - vid), pp=True)} / mean draw "
                                     f"{fmt(P.rate(mean_draw[h] - vid), pp=True)}")
                    print(f"    v2 >= {cut}: " + " | ".join(parts))
                if suffix:
                    for cut in (0.9, 0.7):
                        vid = (v2 >= cut).astype(float)
                        local = np.array([r[f"{m}_layout_correct"] == "1" for r in rows], float)
                        print(f"    local vLLM layout, v2 >= {cut}: {fmt(P.rate(local - vid), pp=True)}")


if __name__ == "__main__":
    main()
