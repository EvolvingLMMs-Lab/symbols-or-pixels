# Results

Tables 2–3, Figures 1–4, and the paired results of the note can be recomputed from the files below. The 1024² text runs, the training-overlap check, and the API-served 27B run are not included.
- `analysis/build_results.py` builds `per_sample.csv` and `excluded_tasks.csv` from `raw/`;
- `analysis/analyze.py` builds `summary/` from them.

Systems are named as follows:
- video models: `g5` (VBVR-Pro-Wan2.2-TI2V-5B) and `g27` (VBVR-Pro-Wan2.2-I2V-A14B);
- text systems: `<model>_<prompt>`, with model `qwen3.5-4b`, `qwen3.5-9b` or `qwen3.6-27b` and prompt `layout` or `direct`.

## `per_sample.csv`

One row per paired sample: 475 rows (95 tasks × 5 samples), sorted by task name and sample index.

| Column | Description |
| --- | --- |
| `task` | Benchmark task directory name, e.g. `G-131_select_next_figure_increasing_size_sequence_data-generator` |
| `task_id` | Short task id, e.g. `G-131` |
| `idx` | Sample index within the task (0–4), the `0000<idx>` directory of the benchmark |
| `split` | `in_domain` (task family seen in VBVR-Pro training, `In-Domain_50`) or `out_of_domain` (`Out-of-Domain_50`) |
| `task_class` | `selection_marking`, `multi_object_discrete`, `single_agent_path` or `continuous`, from `specs_v1.json` |
| `answer_kind` | `pixel` when the answer contains pixel positions (tasks listed in `px_fields_v1.json`), else `non_pixel` |
| `g5_v2`, `g27_v2` | VBVR-Pro-Bench v2 score of the generated video, in [0, 1]; solved at ≥ 0.9 (strict) or ≥ 0.7 (lenient) |
| `<system>_correct` | 1 when the parsed final answer is correct, else 0 |
| `<system>_outcome` | `correct`; `wrong` (parsed but wrong); `no_answer` (no parsable `ANSWER:` line); `truncated` (no parsable answer and the output hit the 2,048-token cap) |
| `<system>_prompt_tokens` | Prompt tokens of the call, including the image tokens |
| `<system>_completion_tokens` | Generated tokens |
| `<system>_flops` | Forward FLOPs of the call (`analysis/flops.py`) |
| `<system>_answer` | Parsed `ANSWER:` object as JSON, in the coordinates of the 512 × 512 frame the model saw (`null` if none) |

## `excluded_tasks.csv`

The five tasks outside the paired set, with the reason for each.
- G-21, O-55 and O-9 have no consistent final answer. The detail column quotes `specs_v1.json`.
- O-33 and G-161 fail a check of the video scorer:
  - O-33: 3 of its 5 ground-truth videos score 0 (see `raw/video/ground_truth_v2_scores.jsonl`).
  - G-161: a video that does nothing receives full marks.

## `raw/`

- `raw/text/<system>/responses.jsonl` holds the model outputs, one line per call. There are 485 calls: the 97 tasks with an answer specification × 5 samples. The fields are:
  - `model`, `hint` (the prompt condition), `task`, `split`, `idx`;
  - `input_px` and `eval_size`, the frame size sent, 512;
  - `content`, the raw output;
  - `finish`, which is `stop` or `length`;
  - `provider`, the vLLM version;
  - `usage`, the token counts.
- `raw/text/<system>/graded.jsonl` is the output of `inference/text/run_zeroshot.py grade`. It holds the parsed `answer`, `correct`, the comparator detail `why`, the tokens, and `px_scale`. `px_scale` is native pixels per pixel of the frame sent, so 2.0 for the 512 frame.
- `raw/text/<system>/run_info.json` holds the model, vLLM version, prompt condition, frame size, request count and total tokens.
- `raw/video/g5_v2_scores.jsonl` and `raw/video/g27_v2_scores.jsonl` hold the v2 score of every benchmark sample: 500 lines of `split`, `task`, `idx`, `score` and `generation`.
  - `generation` is `seed42` for videos generated with seed 42 and the negative prompt of VBVR-Pro's `example.py`.
  - It is `eval_script` for videos from VBVR-Pro's evaluation script, which covers G5's in-domain samples.
- `raw/video/ground_truth_v2_scores.jsonl` holds the v2 scores of the benchmark's ground-truth videos under the same scorer. It is the basis of the O-33 exclusion.

## `summary/`

`summary.json` holds every table below, with rates and differences as fractions (× 100 = percentage points) rounded to 4 decimals. Each table also has a CSV:

| File | Content (note reference) |
| --- | --- |
| `table2_reproduction.csv` | Mean v2 score over all 500 samples, overall / in-domain / out-of-domain, against the model cards (Table 2) |
| `systems.csv` | Per system: forward FLOPs per answer, solve rate and 95% CI at v2 ≥ 0.9 and ≥ 0.7, and the share of calls that hit the token cap (Figure 1) |
| `table3_paired.csv` | Per pair and prompt: FLOPs, FLOPs ratio, ratio per solved answer, paired differences with CIs (Table 3) |
| `paired_groups.csv` | Per pair, prompt, threshold and group: rates, differences, the union (oracle) rate, and the outcome cells `both`, `text_only`, `video_only` and `neither` (Figures 2 and 3) |
| `text_outcomes.csv` | Outcome counts per text system over the 475 paired samples, overall and by task class (Figure 4, left) |
| `video_scores.csv` | v2 score histogram in bins of 0.1 and solved / near-miss / below counts per video model (Figure 4, right) |

The conventions are:
- **Solve rate.** Rates are task-weighted: each task's success rate over its 5 samples, averaged over tasks.
- **Text success.** A text answer's success does not depend on the video threshold.
- **Differences.** Differences are paired by sample and reported as text − video.
- **Confidence intervals.** The 95% intervals come from a task bootstrap: 4,000 resamples with `random.Random(0)`, a fresh generator per statistic, and nearest-rank percentiles.
- **Text FLOPs per answer.** This value averages over all 485 calls of a run, as in the note.
- **Union.** The union rate counts a sample as solved when either side solves it. It is an oracle upper bound, not a measured ensemble.
