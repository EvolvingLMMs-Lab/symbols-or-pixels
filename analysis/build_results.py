#!/usr/bin/env python3
"""Build the per-sample table of the 475 paired samples from the raw text and video records.

Inputs (all in this repository):
    results/raw/text/<system>/graded.jsonl    graded language-model answers, 485 per system (97 tasks x 5)
    results/raw/video/{g5,g27}_v2_scores.jsonl VBVR-Pro-Bench v2 scores, 500 per model (100 tasks x 5)
    inference/text/specs_v1.json              frozen answer specs (task class, split, 3 excluded tasks)
    inference/text/px_fields_v1.json          tasks whose answers are pixel positions

Paired set: the 97 tasks with an answer spec, minus O-33 and G-161, which fail a check of the video scorer
(O-33: 3 of 5 ground-truth videos score 0; G-161: a video that does nothing receives full marks).

Outputs:
    results/per_sample.csv       one row per paired sample; columns are documented in results/README.md
    results/excluded_tasks.csv   the five excluded tasks and why

Usage:
    python3 analysis/build_results.py [--root .]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flops import text_call_flops  # noqa: E402

TEXT_SYSTEMS = [f"{m}_{p}" for m in ("qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b") for p in ("layout", "direct")]
VIDEO_SYSTEMS = ("g5", "g27")
SCORER_CHECK = {
    "O-33": "scorer check: 3 of 5 ground-truth videos score 0 under the v2 scorer",
    "G-161": "scorer check: a video that does nothing receives full marks",
}


def read_jsonl(path):
    if not path.exists():
        sys.exit(f"missing input {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outcome(g):
    """correct, or why an answer failed: no final answer within the 2,048-token cap (truncated), no parsable
    ANSWER line (no_answer), or a parsed but wrong answer (wrong)."""
    if g["correct"]:
        return "correct"
    no_answer = str(g.get("why", "")).startswith("no answer")
    if no_answer and g.get("finish") == "length":
        return "truncated"
    return "no_answer" if no_answer else "wrong"


def task_id(task):
    return task.split("_", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="repository root")
    args = ap.parse_args()
    root = Path(args.root)
    specs = json.loads((root / "inference/text/specs_v1.json").read_text())
    px = json.loads((root / "inference/text/px_fields_v1.json").read_text())
    pixel_tasks = set(px["all"]) | set(px["keys"])
    # sorted by task name: the bootstrap in analyze.py resamples task positions in this order
    paired = sorted((s for s in specs if s.get("include", True) and task_id(s["task"]) not in SCORER_CHECK),
                    key=lambda s: s["task"])

    video = {}
    for v in VIDEO_SYSTEMS:
        rows = read_jsonl(root / f"results/raw/video/{v}_v2_scores.jsonl")
        video[v] = {(r["task"], r["idx"]): r["score"] for r in rows}
        if len(video[v]) != 500:
            sys.exit(f"{v}: expected 500 scored videos, found {len(video[v])}")
    text = {}
    for s in TEXT_SYSTEMS:
        rows = read_jsonl(root / f"results/raw/text/{s}/graded.jsonl")
        text[s] = {(r["task"], r["idx"]): r for r in rows}
        if len(text[s]) != 485:
            sys.exit(f"{s}: expected 485 graded answers, found {len(text[s])}")

    cols = ["task", "task_id", "idx", "split", "task_class", "answer_kind", "g5_v2", "g27_v2"]
    for s in TEXT_SYSTEMS:
        cols += [f"{s}_{c}" for c in ("correct", "outcome", "prompt_tokens", "completion_tokens", "flops", "answer")]
    out_rows = []
    for spec in paired:
        for idx in range(5):
            key = (spec["task"], idx)
            row = dict(task=spec["task"], task_id=task_id(spec["task"]), idx=idx,
                       split="in_domain" if spec["split"] == "In-Domain_50" else "out_of_domain",
                       task_class=spec["class"], answer_kind="pixel" if spec["task"] in pixel_tasks else "non_pixel",
                       g5_v2=video["g5"][key], g27_v2=video["g27"][key])
            for s in TEXT_SYSTEMS:
                g = text[s][key]
                row.update({f"{s}_correct": int(g["correct"]), f"{s}_outcome": outcome(g),
                            f"{s}_prompt_tokens": g["prompt_tokens"], f"{s}_completion_tokens": g["completion_tokens"],
                            f"{s}_flops": text_call_flops(s.rsplit("_", 1)[0], g["prompt_tokens"],
                                                          g["completion_tokens"]),
                            f"{s}_answer": json.dumps(g["answer"])})
            out_rows.append(row)
    with (root / "results/per_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)

    excluded = [dict(task_id=task_id(s["task"]), task=s["task"], split=s["split"], reason="no consistent final answer",
                     detail=s["exclude_reason"]) for s in specs if not s.get("include", True)]
    excluded += [dict(task_id=t, task=next(s["task"] for s in specs if task_id(s["task"]) == t),
                      split=next(s["split"] for s in specs if task_id(s["task"]) == t), reason="video scorer check",
                      detail=why) for t, why in SCORER_CHECK.items()]
    with (root / "results/excluded_tasks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "task", "split", "reason", "detail"], lineterminator="\n")
        w.writeheader()
        w.writerows(excluded)
    print(f"wrote {len(out_rows)} paired samples ({len(paired)} tasks) and {len(excluded)} excluded tasks")


if __name__ == "__main__":
    main()
