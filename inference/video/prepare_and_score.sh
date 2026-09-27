#!/usr/bin/env bash
# Prepare generated videos the way VBVR-Pro does, score them with the VBVR-Pro-Bench v2 evaluator, and write the
# per-sample scores in this repository's format (results/raw/video/*_v2_scores.jsonl).
#
# Steps:
#   1. VBVR-Pro rl_training/src/cli/prepare_vbvr_eval_videos.py: resize without cropping, pad to a 1024 x 1024
#      canvas, and raise the frame rate so each video lasts at most 5 s (every frame is kept).
#   2. VBVR-Pro-Bench run_evaluation_video.py at 95265d0: task-specific scorers only; one score in [0, 1] per video.
#   3. collect_scores.py: {name}_vbvr_results.json -> one {"split", "task", "idx", "score"} line per video.
#
# Config (env, all required):
#   VBVR_PRO_DIR        checkout of https://github.com/Video-Reason/VBVR-Pro at f613643f01be (with rl_training/.venv)
#   VBVR_PRO_BENCH_DIR  checkout of https://github.com/Video-Reason/VBVR-Pro-Bench at 95265d088437 (deps installed)
#   BENCH_DIR           the VBVR-Pro-Bench-Video data tree (ground truth videos and metadata)
#
# Usage:
#   inference/video/prepare_and_score.sh <generated video dir> <work dir> <name>
#   e.g. inference/video/prepare_and_score.sh videos/g27 work g27  ->  work/g27_v2_scores.jsonl
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <generated video dir> <work dir> <name>" >&2
  exit 2
fi
: "${VBVR_PRO_DIR:?set VBVR_PRO_DIR to a VBVR-Pro checkout}"
: "${VBVR_PRO_BENCH_DIR:?set VBVR_PRO_BENCH_DIR to a VBVR-Pro-Bench checkout}"
: "${BENCH_DIR:?set BENCH_DIR to the VBVR-Pro-Bench-Video data directory}"

raw="$(cd "$1" && pwd)"
mkdir -p "$2"
work="$(cd "$2" && pwd)"
name="$3"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

expected="$(find "$raw" -name '*.mp4' | wc -l | tr -d ' ')"
echo "preparing $expected videos from $raw"
(cd "$VBVR_PRO_DIR/rl_training" && .venv/bin/python -m src.cli.prepare_vbvr_eval_videos \
  --input-dir "$raw" --output-dir "$work/prepared/$name" \
  --width 1024 --height 1024 --max-duration 5 --expected-videos "$expected")

(cd "$VBVR_PRO_BENCH_DIR" && python run_evaluation_video.py \
  --model_path "$work/prepared/$name" --gt_base "$BENCH_DIR" --output_dir "$work/eval")

python3 "$here/collect_scores.py" --results "$work/eval/${name}_vbvr_results.json" --out "$work/${name}_v2_scores.jsonl"
