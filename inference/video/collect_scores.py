#!/usr/bin/env python3
"""Convert a VBVR-Pro-Bench `{name}_vbvr_results.json` into per-sample v2 scores in this repository's format.

Each output line is {"split": "In-Domain_50", "task": <task name>, "idx": <int>, "score": <float>}, sorted by split,
task and idx, the format of results/raw/video/*_v2_scores.jsonl. A missing video scores 0 in the evaluator; such
samples, and samples with an evaluator error, are reported on stderr.

Usage:
    python3 collect_scores.py --results <eval dir>/<name>_vbvr_results.json --out <name>_v2_scores.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.results)
    if not path.exists():
        sys.exit(f"missing evaluator output {path}")
    samples = json.loads(path.read_text())["samples"]
    rows, problems = [], 0
    for s in samples:
        rows.append(dict(split=s["folder"], task=s["task_name"], idx=int(Path(s["video_file"]).stem),
                         score=s["score"]))
        if s.get("error") or s.get("video_path") is None:
            problems += 1
            print(f"{s['folder']}/{s['task_name']}/{s['video_file']}: {s.get('error') or 'no video'}", file=sys.stderr)
    rows.sort(key=lambda r: (r["split"], r["task"], r["idx"]))
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} scores to {args.out}" + (f"; {problems} missing or failed" if problems else ""))


if __name__ == "__main__":
    main()
