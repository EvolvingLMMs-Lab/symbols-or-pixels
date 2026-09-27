# Inference and grading

## Language models (`text/`)

This code produced the released answers and grades. Three things were removed: the API client, the timing runs, and the prompt variants that the note does not use.

Two checks confirm the code matches those runs:
- regrading all 2,910 released answers with `run_zeroshot.py grade` reproduces `results/raw/text/*/graded.jsonl` exactly;
- `vllm_generate.py --dry-run` rebuilds the prompts shown in the note byte for byte.

**Prompt.**
- The model sees the first frame, resized from 1,024 to 512 × 512 (bicubic), followed by this text:

  ```
  The image is the first frame (512x512 pixels) of a visual reasoning task.
  Task: <prompt.txt>

  Do not draw or describe a video. Work out the result the task asks for.
  <instruction>
  Answer format: <answer_format of the task's spec, with 1024x1024 restated as 512x512>
  Finish with one line: ANSWER: {"<answer_key>": ...} (valid JSON on that line).
  ```

- The `<instruction>` depends on the prompt condition:
  - `layout` (primary condition): "First describe the scene as a JSON layout: every relevant object with its shape, colour, pixel position [x, y] (origin top-left) and size, or its grid cell. Then reason step by step over that layout. Then give the ANSWER line."
  - `direct`: "Give only the final ANSWER line, with no reasoning."

**Decoding.** vLLM 0.29.0 in BF16, greedy decoding with seed 0, a 2,048-token cap, and thinking disabled through the chat template (`enable_thinking=False`).

**Grading.** `run_zeroshot.py grade` grades each answer in four steps:
1. It parses the JSON after the last `ANSWER:` line.
2. It multiplies pixel values by 2 to return to the native 1,024 frame. `px_fields_v1.json` lists the tasks and fields that hold pixel values.
3. It compares the result with the sample's `metadata.json` using the comparator named in `specs_v1.json`.
4. It marks an unparsable answer as incorrect and records the reason.

**Answer specifications (`specs_v1.json`).** The specifications were frozen before any answer was graded. There is one entry per benchmark task:

| Field | Meaning |
| --- | --- |
| `task`, `split`, `class` | Task directory, benchmark split, task class |
| `include`, `exclude_reason` | `false` for the 3 tasks with no consistent final answer (G-21, O-55, O-9), with the reason |
| `answer_key`, `answer_format`, `answer_example` | The key of the `ANSWER` object and the format stated to the model |
| `comparator` | Comparator type and its metadata paths (`sgt.*` = semantic ground truth, `par.*` = parameters) and tolerances. The types are exact, tolerance, nearest-candidate, grid and graph path, move-sequence, and 24 hand-written graders in `custom_graders.py`. |
| `prompt_specified` | `true` when the task prompt already states the rule or target that fixes the answer, so that the image is needed mainly to locate objects |
| `notes` | How the comparator was checked against the rendered frames |

`grader.py check` runs the known-answer check on every included sample. The ground-truth answer must pass and a plausible wrong answer must fail. All 97 included tasks pass.

## Video models (`video/`)

These scripts are thin wrappers written for the release. The official code does the work:
- VBVR-Pro `example.py` for generation;
- VBVR-Pro `rl_training/src/cli/prepare_vbvr_eval_videos.py` for preparation;
- VBVR-Pro-Bench `run_evaluation_video.py`, task-specific scorers only, for scoring.

They were not the launcher that produced the released scores. See "Provenance of the released scores" in the top-level README.

| Script | What it does |
| --- | --- |
| `generate_videos.py` | Loads a VBVR-Pro Wan2.2 checkpoint once and writes `<split>/<task>/<idx>.mp4`. Settings: 512 × 512, 81 frames, 50 steps, CFG 5, seed 42, the negative prompt of `example.py`. |
| `prepare_and_score.sh` | Scales each video to 1024 × 1024 without cropping and raises the frame rate so it lasts at most 5 s. Then it runs the v2 evaluator. |
| `collect_scores.py` | Converts the evaluator's `{name}_vbvr_results.json` into `{split, task, idx, score}` lines |
