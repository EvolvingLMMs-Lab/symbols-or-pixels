#!/usr/bin/env python3
"""Compute every summary table of the note from the per-sample results.

Conventions:
    solve rate   task-weighted: the mean over tasks of each task's success rate over its 5 samples
    video        solved when the v2 score is >= the threshold (strict 0.9, lenient 0.7)
    text         solved when the graded final answer is correct (the same at both thresholds)
    difference   paired by sample, text - video, as a fraction (x 100 = percentage points)
    95% CI       task bootstrap: 4,000 resamples of the task list with random.Random(0) (a fresh generator per
                 statistic, n draws of randrange(n) per resample), nearest-rank 2.5th and 97.5th percentiles
    FLOPs        forward FLOPs per answer (analysis/flops.py); for text, the mean over all 485 calls of a run
                 (97 tasks x 5), as in the note; for video, one generated video

Inputs: results/per_sample.csv (analysis/build_results.py), results/raw/text/<system>/graded.jsonl (FLOPs and
truncation over all calls) and results/raw/video/*_v2_scores.jsonl (reproduction means over all 500 samples).
Outputs: results/summary/summary.json and one CSV per table (see results/README.md).

Usage:
    python3 analysis/analyze.py [--root .]
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_results import TEXT_SYSTEMS, VIDEO_SYSTEMS, read_jsonl  # noqa: E402
from flops import TEXT_MODELS, VIDEO_MODELS, text_call_flops, video_flops  # noqa: E402

B, SEED = 4000, 0
CUTS = ("0.9", "0.7")
PROMPTS = ("layout", "direct")
PAIRS = {"5B": ("g5", "qwen3.5-4b"), "27B": ("g27", "qwen3.6-27b")}
LABELS = {"g5": "G5 (VBVR-Pro-Wan2.2-TI2V-5B)", "g27": "G27 (VBVR-Pro-Wan2.2-I2V-A14B)",
          **{s: f"{TEXT_MODELS[s.rsplit('_', 1)[0]]['model'].split('/')[1]}, {s.rsplit('_', 1)[1]} prompt"
             for s in TEXT_SYSTEMS}}
CLASSES = ("selection_marking", "multi_object_discrete", "single_agent_path", "continuous")
GROUPS = [("overall", "All tasks", "Split", lambda r: True),
          ("in_domain", "In-domain", "Split", lambda r: r["split"] == "in_domain"),
          ("out_of_domain", "Out-of-domain", "Split", lambda r: r["split"] == "out_of_domain"),
          ("selection_marking", "Selection / marking", "Task class", lambda r: r["task_class"] == "selection_marking"),
          ("multi_object_discrete", "Multi-object discrete", "Task class",
           lambda r: r["task_class"] == "multi_object_discrete"),
          ("single_agent_path", "Single-agent path", "Task class", lambda r: r["task_class"] == "single_agent_path"),
          ("continuous", "Continuous", "Task class", lambda r: r["task_class"] == "continuous"),
          ("pixel_answer", "Pixel-position answer", "Answer kind", lambda r: r["answer_kind"] == "pixel"),
          ("other_answer", "Non-pixel answer", "Answer kind", lambda r: r["answer_kind"] == "non_pixel")]
PUBLISHED_V2 = {"g5": (0.470, 0.641, 0.300), "g27": (0.670, 0.808, 0.532)}  # model cards: overall / ID / OOD


@lru_cache(maxsize=None)
def boot_indices(n):
    rng = random.Random(SEED)
    return np.array([[rng.randrange(n) for _ in range(n)] for _ in range(B)])


def ci(per_task):
    vals = per_task[boot_indices(len(per_task))].mean(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5], method="nearest")
    return [round(float(lo), 4), round(float(hi), 4)]


def solved(row, system, cut):
    if system in VIDEO_SYSTEMS:
        return float(row[f"{system}_v2"]) >= float(cut)
    return row[f"{system}_correct"] == "1"


def by_task(rows):
    tasks = defaultdict(list)
    for r in rows:
        tasks[r["task"]].append(r)
    return list(tasks.values())  # insertion order = per_sample.csv order (task name)


def per_task_rate(tasks, fn):
    return np.array([sum(fn(r) for r in t) / len(t) for t in tasks])


def rate_ci(tasks, fn):
    x = per_task_rate(tasks, fn)
    return dict(rate=round(float(x.mean()), 4), ci=ci(x), tasks=len(tasks))


def paired_group(rows, video, text, cut):
    tasks = by_task(rows)
    t = per_task_rate(tasks, lambda r: solved(r, text, cut))
    v = per_task_rate(tasks, lambda r: solved(r, video, cut))
    u = per_task_rate(tasks, lambda r: solved(r, text, cut) or solved(r, video, cut))
    cells = Counter(("both" if a and b else "text_only" if a else "video_only" if b else "neither")
                    for r in rows for a, b in [(solved(r, text, cut), solved(r, video, cut))])
    return dict(tasks=len(tasks), text=dict(rate=round(float(t.mean()), 4), ci=ci(t)),
                video=dict(rate=round(float(v.mean()), 4), ci=ci(v)), union=dict(rate=round(float(u.mean()), 4), ci=ci(u)),
                diff=round(float(t.mean() - v.mean()), 4), ci=ci(t - v),
                cells={k: cells.get(k, 0) for k in ("both", "text_only", "video_only", "neither")})


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="repository root")
    args = ap.parse_args()
    root = Path(args.root)
    path = root / "results/per_sample.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run analysis/build_results.py first")
    rows = list(csv.DictReader(path.open()))
    tasks = by_task(rows)
    out = root / "results/summary"
    out.mkdir(parents=True, exist_ok=True)

    # Table 2: reproduction of the published mean v2 scores, over all 100 tasks x 5 samples
    table2 = []
    for v in VIDEO_SYSTEMS:
        scores = read_jsonl(root / f"results/raw/video/{v}_v2_scores.jsonl")
        mean = lambda sel: sum(r["score"] for r in sel) / len(sel)  # noqa: E731
        ours = (mean(scores), mean([r for r in scores if r["split"] == "In-Domain_50"]),
                mean([r for r in scores if r["split"] == "Out-of-Domain_50"]))
        table2.append(dict(model=v, checkpoint=VIDEO_MODELS[v]["checkpoint"], samples=len(scores),
                           ours_overall=round(ours[0], 4), ours_in_domain=round(ours[1], 4),
                           ours_out_of_domain=round(ours[2], 4), published_overall=PUBLISHED_V2[v][0],
                           published_in_domain=PUBLISHED_V2[v][1], published_out_of_domain=PUBLISHED_V2[v][2]))

    # Figure 1 systems: FLOPs per answer and solve rates over the 95 paired tasks
    systems = {}
    for v in VIDEO_SYSTEMS:
        systems[v] = dict(kind="video", label=LABELS[v], flops=video_flops(v),
                          **{f"solve_{c}": rate_ci(tasks, lambda r, c=c: solved(r, v, c)) for c in CUTS})
    for s in TEXT_SYSTEMS:
        calls = read_jsonl(root / f"results/raw/text/{s}/graded.jsonl")
        flops = [text_call_flops(s.rsplit("_", 1)[0], g["prompt_tokens"], g["completion_tokens"]) for g in calls]
        sr = rate_ci(tasks, lambda r, s=s: solved(r, s, "0.9"))  # text success does not depend on the threshold
        systems[s] = {"kind": "text", "label": LABELS[s], "flops": sum(flops) / len(flops), "calls": len(calls),
                      "truncated_share_all_calls": sum(g["finish"] == "length" for g in calls) / len(calls),
                      "solve_0.9": sr, "solve_0.7": sr}

    # Table 3 and Figures 2-3: paired comparisons for every pair, prompt, threshold and group
    table3, groups = [], []
    for pair, (v, m) in PAIRS.items():
        for p in PROMPTS:
            t = f"{m}_{p}"
            ratio = systems[v]["flops"] / systems[t]["flops"]
            entry = dict(pair=pair, video=v, text=t, prompt=p, video_flops=systems[v]["flops"],
                         text_flops=systems[t]["flops"], flops_ratio=ratio)
            for c in CUTS:
                vr = per_task_rate(tasks, lambda r, c=c: solved(r, v, c)).mean()
                tr = per_task_rate(tasks, lambda r: solved(r, t, c)).mean()
                entry[f"ratio_per_solved_{c}"] = ratio * tr / vr
            for c in CUTS:
                for key, label, block, sel in GROUPS:
                    g = paired_group([r for r in rows if sel(r)], v, t, c)
                    groups.append(dict(pair=pair, prompt=p, cut=c, group=key, label=label, block=block, **g))
                    if key == "overall":
                        entry[f"text_rate_{c}"], entry[f"video_rate_{c}"] = g["text"]["rate"], g["video"]["rate"]
                        entry[f"diff_{c}"], entry[f"diff_{c}_ci"] = g["diff"], g["ci"]
            table3.append(entry)

    # Figure 4: text outcomes and video score distributions over the 475 paired samples
    text_outcomes = {}
    for s in TEXT_SYSTEMS:
        count = lambda sel: {k: sum(r[f"{s}_outcome"] == k for r in sel)  # noqa: E731
                             for k in ("correct", "wrong", "no_answer", "truncated")}
        text_outcomes[s] = dict(overall=count(rows),
                                by_class={c: count([r for r in rows if r["task_class"] == c]) for c in CLASSES})
    video_scores = {}
    for v in VIDEO_SYSTEMS:
        sc = [float(r[f"{v}_v2"]) for r in rows]
        hist = [0] * 10
        for x in sc:
            hist[min(int(x * 10), 9)] += 1
        band = lambda sel: dict(solved=sum(float(r[f"{v}_v2"]) >= 0.9 for r in sel),  # noqa: E731
                                near=sum(0.7 <= float(r[f"{v}_v2"]) < 0.9 for r in sel),
                                below=sum(float(r[f"{v}_v2"]) < 0.7 for r in sel))
        video_scores[v] = dict(hist=hist, overall=band(rows),
                               by_class={c: band([r for r in rows if r["task_class"] == c]) for c in CLASSES})

    summary = dict(conventions=__doc__.split("Conventions:")[1].split("Inputs:")[0].strip(), table2=table2,
                   systems=systems, table3=table3, groups=groups, text_outcomes=text_outcomes,
                   video_scores=video_scores)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    write_csv(out / "table2_reproduction.csv", table2)
    write_csv(out / "systems.csv", [dict(system=k, kind=e["kind"], label=e["label"], flops_per_answer=e["flops"],
                                         solve_0_9=e["solve_0.9"]["rate"], solve_0_9_ci_lo=e["solve_0.9"]["ci"][0],
                                         solve_0_9_ci_hi=e["solve_0.9"]["ci"][1], solve_0_7=e["solve_0.7"]["rate"],
                                         solve_0_7_ci_lo=e["solve_0.7"]["ci"][0],
                                         solve_0_7_ci_hi=e["solve_0.7"]["ci"][1],
                                         truncated_share_all_calls=e.get("truncated_share_all_calls", ""))
                                    for k, e in systems.items()])
    write_csv(out / "table3_paired.csv", [dict(pair=e["pair"], prompt=e["prompt"], video=e["video"], text=e["text"],
                                               video_flops=e["video_flops"], text_flops=e["text_flops"],
                                               flops_ratio=e["flops_ratio"],
                                               ratio_per_solved_0_9=e["ratio_per_solved_0.9"],
                                               ratio_per_solved_0_7=e["ratio_per_solved_0.7"],
                                               diff_0_9=e["diff_0.9"], diff_0_9_ci_lo=e["diff_0.9_ci"][0],
                                               diff_0_9_ci_hi=e["diff_0.9_ci"][1], diff_0_7=e["diff_0.7"],
                                               diff_0_7_ci_lo=e["diff_0.7_ci"][0], diff_0_7_ci_hi=e["diff_0.7_ci"][1])
                                          for e in table3])
    write_csv(out / "paired_groups.csv", [dict(pair=g["pair"], prompt=g["prompt"], cut=g["cut"], group=g["group"],
                                               tasks=g["tasks"], text_rate=g["text"]["rate"],
                                               text_ci_lo=g["text"]["ci"][0], text_ci_hi=g["text"]["ci"][1],
                                               video_rate=g["video"]["rate"], video_ci_lo=g["video"]["ci"][0],
                                               video_ci_hi=g["video"]["ci"][1], diff=g["diff"], diff_ci_lo=g["ci"][0],
                                               diff_ci_hi=g["ci"][1], union_rate=g["union"]["rate"],
                                               union_ci_lo=g["union"]["ci"][0], union_ci_hi=g["union"]["ci"][1],
                                               **g["cells"]) for g in groups])
    write_csv(out / "text_outcomes.csv", [dict(system=s, task_class=c, **counts) for s, e in text_outcomes.items()
                                          for c, counts in [("all", e["overall"]), *e["by_class"].items()]])
    write_csv(out / "video_scores.csv", [dict(model=v, **{f"bin_{i / 10:.1f}": n for i, n in enumerate(e["hist"])},
                                              **e["overall"]) for v, e in video_scores.items()])

    print("Paired comparisons (text - video, percentage points, 95% CI):")
    for e in table3:
        print(f"  {e['pair']:>3} {e['prompt']:6s} FLOPs ratio {e['flops_ratio']:7,.0f}x  per solved "
              f"{e['ratio_per_solved_0.9']:6,.0f}x ({e['ratio_per_solved_0.7']:,.0f}x)  "
              f"v2>=0.9 {100 * e['diff_0.9']:+.1f} [{100 * e['diff_0.9_ci'][0]:+.1f}, {100 * e['diff_0.9_ci'][1]:+.1f}]  "
              f"v2>=0.7 {100 * e['diff_0.7']:+.1f} [{100 * e['diff_0.7_ci'][0]:+.1f}, {100 * e['diff_0.7_ci'][1]:+.1f}]")
    print(f"wrote {out}/summary.json and 6 CSV tables")


if __name__ == "__main__":
    main()
