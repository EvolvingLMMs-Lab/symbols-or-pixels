#!/usr/bin/env python3
"""Prompts, answer parsing and grading for the zero-shot language-model runs on VBVR-Pro-Bench.

For every (model, prompt, sample) the model gets first_frame.png + prompt.txt + a prompt-specific instruction +
the task's answer format, and must end with `ANSWER: {json}`. vllm_generate.py builds the prompts with
build_prompt() and writes responses.jsonl; the `grade` command below parses the last ANSWER line of each response
and grades it with grader.py against the sample's metadata.json.

Prompts ("hints" in the record schema):
    layout  describe the scene as a JSON layout, reason over it, then answer (primary condition)
    direct  give only the final ANSWER line

Config: none. Inputs are the frozen answer specs (specs_v1.json), the benchmark tree (Hugging Face dataset
Video-Reason/VBVR-Pro-Bench, video setting) and a run directory that holds responses.jsonl.

Usage:
    python3 run_zeroshot.py grade --specs specs_v1.json --bench <VBVR-Pro-Bench-Video dir> --out <run dir>
"""

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader  # noqa: E402

HINTS = {
    "direct": "Give only the final ANSWER line, with no reasoning.",
    "layout": (
        "First describe the scene as a JSON layout: every relevant object with its shape, colour, pixel "
        "position [x, y] (origin top-left) and size, or its grid cell. Then reason step by step over that "
        "layout. Then give the ANSWER line."
    ),
}

TEMPLATE = """The image is the first frame ({w}x{h} pixels) of a visual reasoning task.
Task: {prompt}

Do not draw or describe a video. Work out the result the task asks for.
{hint}
Answer format: {answer_format}
{finish}"""


def png_size(data):
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def resize_png(data, size):
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def prompt_text(spec, prompt, hint, w, h, frame_w=1024, frame_h=1024):
    """The prompt for a first frame shown at w x h. The answer formats name the bench frame size (frame_w x frame_h,
    1024 x 1024); for any other shown size that phrase is restated at w x h."""
    answer_format = spec["answer_format"]
    if (w, h) != (frame_w, frame_h):
        answer_format = answer_format.replace(f"{frame_w}x{frame_h}", f"{w}x{h}")
    finish = f'Finish with one line: ANSWER: {{"{spec["answer_key"]}": ...}} (valid JSON on that line).'
    return TEMPLATE.format(w=w, h=h, prompt=prompt, hint=HINTS[hint], answer_format=answer_format, finish=finish)


def build_prompt(spec, sample_dir, hint, size=0):
    """Prompt text and base64 PNG. size > 0 (512 in this study, to match the video models' input) sends the frame at
    size x size and states that size, also inside the answer format; pixel answers are scaled back when graded."""
    image = (sample_dir / "first_frame.png").read_bytes()
    w, h = png_size(image)
    if size:
        image, w, h = resize_png(image, size), size, size
    prompt = (sample_dir / "prompt.txt").read_text().strip()
    return prompt_text(spec, prompt, hint, w, h), base64.b64encode(image).decode()


# ---------------------------------------------------------------- answer extraction


def _balanced_json(s, start):
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_answer_line(text):
    idx = text.rfind("ANSWER:")
    if idx < 0:
        raise ValueError("no ANSWER line")
    rest = text[idx + len("ANSWER:"):].strip().strip("`")
    start = min([i for i in (rest.find("{"), rest.find("[")) if i >= 0], default=-1)
    if start < 0:
        return json.loads(rest.splitlines()[0])
    blob = _balanced_json(rest, start)
    return json.loads(blob)


# ---------------------------------------------------------------- commands


def jobs_for(specs, bench, hints, tasks, idxs):
    for spec in specs:
        if not spec.get("include", True) or (tasks and spec["task"] not in tasks):
            continue
        for idx in idxs:
            sample = Path(bench) / spec["split"] / spec["task"] / f"{idx:05d}"
            if sample.exists():
                for hint in hints:
                    yield spec, idx, hint, sample


def read_responses(path):
    """Cached responses; a torn last line (process stopped mid-write) is skipped and reported."""
    if not path.exists():
        sys.exit(f"no responses at {path}")
    rows, lines = [], path.read_text().splitlines()
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise
            print(f"skipping a torn last line in {path}", flush=True)
    return rows


PX_RULES = json.loads((Path(__file__).resolve().parent / "px_fields_v1.json").read_text())


def scale_px(spec, answer_obj, factor):
    """Map pixel values in an answer given on a resized frame back to native first-frame pixels."""
    if factor == 1 or answer_obj is None:
        return answer_obj

    def mul(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v * factor
        if isinstance(v, list):
            return [mul(x) for x in v]
        if isinstance(v, dict):
            return {k: mul(x) for k, x in v.items()}
        return v

    if spec["task"] in PX_RULES["all"]:
        return mul(answer_obj)
    keys = set(PX_RULES["keys"].get(spec["task"], []))
    if not keys:
        return answer_obj

    def walk(v):
        if isinstance(v, dict):
            return {k: (mul(x) if k in keys else walk(x)) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    return walk(answer_obj)


def cmd_grade(args):
    specs = {s["task"]: s for s in json.loads(Path(args.specs).read_text())}
    out = Path(args.out)
    latest = {}
    for r in read_responses(out / "responses.jsonl"):
        if "error" not in r:
            latest[(r["model"], r["hint"], r["task"], r["idx"])] = r
    graded = []
    for (model, hint, task, idx), r in sorted(latest.items()):
        spec = specs.get(task)
        if spec is None or not spec.get("include", True):
            continue
        meta = grader.load_meta(args.bench, spec, idx)
        factor = 1
        if r.get("eval_size"):
            native = png_size((Path(args.bench) / spec["split"] / task / f"{idx:05d}" / "first_frame.png").read_bytes())[0]
            factor = native / r["eval_size"]
        try:
            ans = parse_answer_line(r["content"])
            ok, why = grader.grade(spec, meta, scale_px(spec, ans, factor))
        except Exception as e:
            ans, ok, why = None, False, f"no answer: {type(e).__name__}: {str(e)[:200]}"
        u = r.get("usage") or {}
        graded.append(dict(model=model, hint=hint, task=task, split=r["split"], idx=idx, cls=spec["class"],
                           prompt_specified=spec.get("prompt_specified", False), correct=bool(ok), why=why,
                           answer=ans, finish=r.get("finish"), prompt_tokens=u.get("prompt_tokens"),
                           completion_tokens=u.get("completion_tokens"), eval_size=r.get("eval_size"),
                           px_scale=factor))
    with (out / "graded.jsonl").open("w") as f:
        for g in graded:
            f.write(json.dumps(g, default=str) + "\n")
    summarize(graded, out)


def summarize(graded, out):
    from collections import defaultdict

    def rate(rows):
        by_task = defaultdict(list)
        for g in rows:
            by_task[g["task"]].append(g["correct"])
        per_task = [sum(v) / len(v) for v in by_task.values()]
        return (sum(per_task) / len(per_task) if per_task else float("nan")), len(by_task), len(rows)

    table = {}
    keys = sorted({(g["model"], g["hint"]) for g in graded})
    for model, hint in keys:
        rows = [g for g in graded if g["model"] == model and g["hint"] == hint]
        entry = {}
        for name, sel in [("overall", rows), ("in_domain", [g for g in rows if g["split"] == "In-Domain_50"]),
                          ("out_of_domain", [g for g in rows if g["split"] == "Out-of-Domain_50"])]:
            entry[name] = rate(sel)
        for cls in sorted({g["cls"] for g in rows}):
            entry[f"class:{cls}"] = rate([g for g in rows if g["cls"] == cls])
        comp = [g["completion_tokens"] for g in rows if g["completion_tokens"]]
        entry["completion_tokens_median"] = sorted(comp)[len(comp) // 2] if comp else None
        table[f"{model} | {hint}"] = entry
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    for k, e in table.items():
        o, i, d = e["overall"], e["in_domain"], e["out_of_domain"]
        print(f"{k:40s} overall {o[0]:.3f} ({o[1]} tasks, {o[2]} samples) | ID {i[0]:.3f} | OOD {d[0]:.3f} | "
              f"median out tokens {e['completion_tokens_median']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade", help="parse and grade <out>/responses.jsonl into <out>/graded.jsonl")
    g.add_argument("--specs", required=True)
    g.add_argument("--bench", required=True)
    g.add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_grade(args)


if __name__ == "__main__":
    main()
