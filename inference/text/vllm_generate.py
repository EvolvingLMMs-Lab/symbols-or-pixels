#!/usr/bin/env python3
"""Zero-shot language-model answers on VBVR-Pro-Bench with vLLM.

Each prompt is built with run_zeroshot.build_prompt (task prompt, answer format, and the first frame as a PNG data
URL), generated greedily with seed 0, a 2,048-token decode cap and thinking off through the chat template, then
written to responses.jsonl, which `run_zeroshot.py grade` scores unchanged.

--eval-size N sends the first frame at N x N and states N x N in the prompt (also inside the answer format). The
study uses --eval-size 512, the resolution of the video models' generation. Responses carry eval_size, so
`run_zeroshot.py grade` scales pixel answers back to the native 1024 x 1024 frame before grading.

--manifest (optional) checks the benchmark inputs against a sha256 manifest first; inference/text/bench_inputs.sha256
lists the files this study read.

--dry-run writes prompts.jsonl (the exact prompt text per request) and exits without loading a model.

Config: none beyond the CLI; model weights download into the default Hugging Face cache (HF_HOME if set).

Usage:
    python3 vllm_generate.py --model Qwen/Qwen3.6-27B --specs specs_v1.json --bench <VBVR-Pro-Bench-Video dir> \
        --out <run dir> --hints layout --eval-size 512 [--manifest bench_inputs.sha256] [--idx 0,1,2,3,4] \
        [--tasks T1,T2] [--limit N] [--dry-run]
"""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_zeroshot  # noqa: E402


def check_manifest(bench, manifest):
    bad = []
    for line in Path(manifest).read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = Path(bench) / rel
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    if bad:
        sys.exit(f"bench inputs differ from the manifest in {len(bad)} files, e.g. {bad[:3]}")
    print(f"bench manifest ok: {len(Path(manifest).read_text().splitlines())} files", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--label", default="", help="model name written to responses.jsonl (default: --model)")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--manifest", default="", help="sha256 manifest of the bench inputs to check first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hints", default="layout,direct")
    ap.add_argument("--idx", default="0,1,2,3,4")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N requests only (smoke)")
    ap.add_argument("--eval-size", type=int, default=0, help="send and state the first frame at N x N (study: 512)")
    ap.add_argument("--max-tokens", type=int, default=2048, help="decode cap")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true", help="write prompts.jsonl and exit without loading a model")
    args = ap.parse_args()
    if not Path(args.bench).is_dir():
        sys.exit(f"--bench {args.bench} is not a directory")
    if args.manifest:
        check_manifest(args.bench, args.manifest)
    label = args.label or args.model

    specs = json.loads(Path(args.specs).read_text())
    tasks = set(args.tasks.split(",")) if args.tasks else None
    jobs = list(run_zeroshot.jobs_for(specs, args.bench, args.hints.split(","), tasks,
                                      [int(i) for i in args.idx.split(",")]))
    if not jobs:
        sys.exit(f"no samples found under {args.bench} for the selected tasks and idx")
    if args.limit:
        jobs = jobs[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    msgs, params, recs, texts = [], [], [], []
    for spec, idx, hint, sample in jobs:
        text, img = run_zeroshot.build_prompt(spec, sample, hint, size=args.eval_size)
        sent = run_zeroshot.png_size(base64.b64decode(img))[0]
        msgs.append([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            {"type": "text", "text": text}]}])
        texts.append(text)
        recs.append(dict(model=label, hint=hint, task=spec["task"], split=spec["split"], idx=idx,
                         input_px=sent, eval_size=args.eval_size or None))
    if args.dry_run:
        with (out / "prompts.jsonl").open("w") as f:
            for rec, text in zip(recs, texts):
                f.write(json.dumps(dict(rec, prompt=text)) + "\n")
        print(f"wrote {len(recs)} prompts to {out / 'prompts.jsonl'}", flush=True)
        return

    import vllm
    from vllm import LLM, SamplingParams

    for _ in jobs:
        params.append(SamplingParams(temperature=0, seed=0, max_tokens=args.max_tokens))
    print(f"{len(jobs)} requests for {args.model} on vllm {vllm.__version__}", flush=True)

    llm = LLM(model=args.model, seed=0, max_model_len=args.max_model_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.9, dtype=args.dtype, enable_prefix_caching=True)
    outs = llm.chat(msgs, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
    leaks = 0
    with (out / "responses.jsonl").open("w") as f:
        for rec, o in zip(recs, outs):
            gen = o.outputs[0]
            leaks += "<think>" in gen.text
            rec.update(content=gen.text, finish=gen.finish_reason, provider=f"vllm-{vllm.__version__}",
                       usage={"prompt_tokens": len(o.prompt_token_ids), "completion_tokens": len(gen.token_ids)})
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(outs)} responses; {leaks} contain <think>", flush=True)
    (out / "run_info.json").write_text(json.dumps(dict(
        model=args.model, vllm=vllm.__version__, hints=args.hints, idx=args.idx, eval_size=args.eval_size or None,
        requests=len(outs), prompt_tokens=sum(len(o.prompt_token_ids) for o in outs),
        completion_tokens=sum(len(o.outputs[0].token_ids) for o in outs)), indent=1) + "\n")


if __name__ == "__main__":
    main()
