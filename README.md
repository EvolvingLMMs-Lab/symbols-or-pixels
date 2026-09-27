# Symbols or Pixels?

Code and per-sample results for the LMMs-Lab note [Symbols or Pixels? Zero-shot language reasoning versus VBVR-trained video generation on visual reasoning tasks](https://www.lmms-lab.com/notes/symbols-or-pixels/).

We compare two released VBVR-Pro video models with zero-shot Qwen language models of similar size on 475 paired samples of [VBVR-Pro-Bench](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-Bench) (95 tasks, 5 samples each).
- **Models.** The video models are G5 ([VBVR-Pro-Wan2.2-TI2V-5B](https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.2-TI2V-5B)) and G27 ([VBVR-Pro-Wan2.2-I2V-A14B](https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B)). The language models are Qwen3.5-4B and Qwen3.6-27B, with Qwen3.5-9B as a reference.
- **Input.** Both sides receive the same 512 × 512 first frame and task prompt.
- **Video grading.** A video model generates an 81-frame solution, which the VBVR-Pro-Bench v2 evaluator scores in [0, 1]. A video counts as solved at v2 ≥ 0.9 (strict) or v2 ≥ 0.7 (lenient).
- **Text grading.** A language model writes a final answer, which is graded as correct or incorrect against frozen answer specifications.

This repository contains:
- the inference and grading code;
- the analysis and plotting scripts;
- the raw model outputs and v2 scores;
- a per-sample table from which Tables 2–3, Figures 1–4, and the paired results of the note can be recomputed.

Three results cited in the note are not included: the 1024² text runs, the training-overlap check, and the API-served 27B run.

## Main results

Paired comparisons (Table 3 of the note).
- **FLOPs:** forward FLOPs per answer.
- **Ratio per solved answer:** each side's FLOPs divided by its solve rate. The value is at v2 ≥ 0.9, with v2 ≥ 0.7 in parentheses.
- **Differences:** text − video solve rates in percentage points, with 95% task-bootstrap confidence intervals.

| Pair | Text prompt | Video FLOPs | Text FLOPs | FLOPs ratio | Ratio per solved answer | Difference, v2 ≥ 0.9 | Difference, v2 ≥ 0.7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G5 / Qwen3.5-4B | layout | 5.8×10<sup>15</sup> | 1.2×10<sup>13</sup> | 480× | 597× (356×) | +5.5 [−4.0, +14.9] | −9.7 [−20.2, +0.6] |
| G5 / Qwen3.5-4B | direct | 5.8×10<sup>15</sup> | 1.2×10<sup>13</sup> | 482× | 714× (425×) | +10.7 [+0.6, +20.4] | −4.4 [−15.4, +6.1] |
| G27 / Qwen3.6-27B | layout | 9.0×10<sup>16</sup> | 8.8×10<sup>13</sup> | 1,026× | 1,147× (796×) | +4.8 [−6.5, +16.4] | −13.3 [−24.6, −1.9] |
| G27 / Qwen3.6-27B | direct | 9.0×10<sup>16</sup> | 7.0×10<sup>13</sup> | 1,293× | 1,585× (1,100×) | +9.3 [−2.3, +20.6] | −8.8 [−20.6, +2.7] |

Reproduction of the published mean v2 scores, over all 500 benchmark samples (Table 2 of the note):

| Model | Published (model card), overall / in-domain / out-of-domain | This study |
| --- | --- | --- |
| G5 | 0.470 / 0.641 / 0.300 | 0.474 / 0.640 / 0.308 |
| G27 | 0.670 / 0.808 / 0.532 | 0.652 / 0.773 / 0.531 |

The other tables and figures are in [`results/summary/`](results/summary) and [`figures/`](figures). All of them are computed from [`results/per_sample.csv`](results/per_sample.csv), and the columns are described in [`results/README.md`](results/README.md).

## Repository layout

```
inference/
  text/              zero-shot language-model runs: prompts, vLLM generation, answer parsing and grading
    vllm_generate.py   build prompts and generate answers with vLLM (greedy, 2,048-token cap, thinking off)
    run_zeroshot.py    prompt text, ANSWER-line parsing, `grade` command
    grader.py          comparators, known-answer check (`check` command)
    custom_graders.py  hand-written graders for 24 tasks
    specs_v1.json      frozen answer specifications (answer key, format, comparator) for all 100 tasks
    px_fields_v1.json  answer fields that are pixel values, rescaled from the 512 frame before grading
    bench_inputs.sha256  sha256 of the benchmark inputs this study read
  video/             thin wrappers around the official VBVR-Pro and VBVR-Pro-Bench code
    generate_videos.py   one video per sample with a VBVR-Pro Wan2.2 checkpoint
    prepare_and_score.sh VBVR-Pro video preparation, then the v2 evaluator
    collect_scores.py    evaluator JSON -> per-sample scores
analysis/
  build_results.py   raw records -> results/per_sample.csv, results/excluded_tasks.csv
  analyze.py         per-sample table -> results/summary/ (all tables of the note)
  flops.py           forward-FLOPs model for videos and language-model calls
  plot.py            static Figures 1-4 -> figures/
results/
  per_sample.csv     475 paired samples: v2 scores, text answers, grades, outcomes, tokens, FLOPs
  excluded_tasks.csv the 5 benchmark tasks left out of the paired set, and why
  raw/text/          raw outputs and grades per language model and prompt (485 answers each)
  raw/video/         v2 scores per video model, and of the ground-truth videos (500 each)
  summary/           summary.json and one CSV per table or figure
figures/             PNG versions of Figures 1-4
```

## Setup

```bash
uv sync                    # analysis and plotting
uv sync --extra inference  # plus vLLM 0.29.0 for the language-model runs
```

## Recompute the tables and figures

No GPU is needed. These commands rebuild everything from the released raw records:

```bash
uv run python analysis/build_results.py   # results/per_sample.csv, results/excluded_tasks.csv
uv run python analysis/analyze.py         # results/summary/*, prints Table 3
uv run python analysis/plot.py            # figures/*.png (default: 27B pair, layout prompt, v2 >= 0.9)
uv run python analysis/plot.py --pair 5B --prompt direct --cut 0.7   # another configuration of Figures 2-3
```

## Re-run the language models

Download the benchmark (video setting), then check that the answer specifications grade every ground-truth answer as correct and a plausible wrong answer as incorrect:

```bash
hf download Video-Reason/VBVR-Pro-Bench --repo-type dataset --local-dir data/VBVR-Pro-Bench
tar xzf data/VBVR-Pro-Bench/VBVR-Pro-Bench-Video.tar.gz -C data/VBVR-Pro-Bench
BENCH=data/VBVR-Pro-Bench/VBVR-Pro-Bench-Video
uv run python inference/text/grader.py check --specs inference/text/specs_v1.json --bench $BENCH
```

Generate and grade the answers. Each run writes `responses.jsonl`, `run_info.json`, and then `graded.jsonl`:

```bash
for model in Qwen3.5-4B Qwen3.5-9B Qwen3.6-27B; do
  for prompt in layout direct; do
    out=runs/$(echo $model | tr A-Z a-z)_$prompt
    uv run python inference/text/vllm_generate.py --model Qwen/$model --specs inference/text/specs_v1.json \
      --bench $BENCH --manifest inference/text/bench_inputs.sha256 --hints $prompt --eval-size 512 --out $out
    uv run python inference/text/run_zeroshot.py grade --specs inference/text/specs_v1.json --bench $BENCH --out $out
  done
done
```

- **Outputs.** Copy a run directory over `results/raw/text/<model>_<prompt>/` and rerun the three analysis commands.
- **Manifest check.** `--manifest` stops the run if the downloaded inputs differ from the ones used here. Drop the flag to run on a newer release of the benchmark.
- **Prompts only.** `--dry-run` writes the exact prompts to `prompts.jsonl` without loading a model.
- **Regrading.** `run_zeroshot.py grade` on the released `results/raw/text/*/responses.jsonl` reproduces the released `graded.jsonl` files exactly.

## Re-run the video models

The video side uses the official code at the revisions this study used:

```bash
git clone https://github.com/Video-Reason/VBVR-Pro && git -C VBVR-Pro checkout f613643f01be6512ad07cdf28213da726e2c676e
git clone https://github.com/Video-Reason/VBVR-Pro-Bench && git -C VBVR-Pro-Bench checkout 95265d088437db6f1f9511552fce44bfd425569c
# install each repository's environment as its README describes (VBVR-Pro: uv; VBVR-Pro-Bench: pip -r requirements.txt)

export VBVR_PRO_DIR=$PWD/VBVR-Pro VBVR_PRO_BENCH_DIR=$PWD/VBVR-Pro-Bench BENCH_DIR=$BENCH
$VBVR_PRO_DIR/.venv/bin/python inference/video/generate_videos.py \
  --model Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B --bench $BENCH --out videos/g27
inference/video/prepare_and_score.sh videos/g27 work g27   # -> work/g27_v2_scores.jsonl
```

The generation settings are:
- 512 × 512, 81 frames;
- 50 denoising steps, classifier-free guidance 5;
- seed 42, 16 fps.

Before scoring, VBVR-Pro's preparation script scales each video to 1024 × 1024 without cropping (padding if the aspect ratio differs) and raises its frame rate so it lasts at most 5 s. Every frame is kept.

**Provenance of the released scores.** The video scripts in this repository are thin wrappers written for the release. They were not the launcher that produced the released scores.
- **G27 and G5 out-of-domain.** These videos used the wrapper's settings: seed 42 and the negative prompt of VBVR-Pro's `example.py`.
- **G5 in-domain.** These videos came from VBVR-Pro's evaluation script, `rl_training/src/cli/eval_i2v.py`. That script uses its own negative prompt, the seed plus the sample index, and a LANCZOS first-frame resize.
- **Per-row record.** Each line of `results/raw/video/g5_v2_scores.jsonl` records which of the two it is.

Regenerated videos are not expected to be bit-identical.

## Notes on the protocol

- **Paired set.** 95 of the 100 benchmark tasks are paired. Five are excluded: G-21, O-55 and O-9 have no consistent final answer, O-33 fails the scorer check with its ground-truth videos, and G-161 gives full marks to a video that does nothing. See [`results/excluded_tasks.csv`](results/excluded_tasks.csv).
- **Solve rates.** Solve rates are task-weighted.
- **Confidence intervals.** The 95% intervals resample tasks: 4,000 resamples with `random.Random(0)`, nearest-rank percentiles.
- **Forward FLOPs.** FLOPs follow the model shapes, and for text also the measured token counts; see [`analysis/flops.py`](analysis/flops.py).
  - Text FLOPs per answer average over all 485 calls of a run (97 tasks), as in the note.
  - The video estimate omits the text encoder and VAE.
  - Text decoding is memory-bound, so wall-clock ratios on real hardware are smaller than the FLOPs ratios.
- **Training and grading differ.** The video models were fine-tuned on VBVR-Pro-SFT and receive a trajectory score with partial credit. The language models are zero-shot and graded on the final answer only. The comparison is between systems as released and does not isolate the effect of representation.

## Citation

```bibtex
@misc{li2026symbols,
  author       = {Li, Brian},
  title        = {Symbols or Pixels? Zero-Shot Language Reasoning versus VBVR-Trained Video Generation on Visual Reasoning Tasks},
  year         = {2026},
  howpublished = {LMMS Lab Notes},
  url          = {https://www.lmms-lab.com/notes/symbols-or-pixels/}
}
```

## License

The code is released under the Apache License 2.0; see [LICENSE](LICENSE). VBVR-Pro-Bench data is licensed separately under CC BY-NC 4.0. This repository contains no benchmark frames or videos.
