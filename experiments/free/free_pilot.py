#!/usr/bin/env python3
"""Free-reasoning pilot: the layout, direct and a neutral "free" instruction, served by one chat-completions endpoint.

Each call sends the same 512 x 512 first frame and prompt template as the zero-shot runs
(inference/text/run_zeroshot.build_prompt); only the instruction line differs. The free instruction is
    "Think it through in whatever way you find useful, then give the ANSWER line."
Draws: `greedy` is temperature 0 with seed 0; `s<N>` samples with temperature 0.7, top_p 0.8, top_k 20, no presence
penalty and seed N. Every call uses an 8,192-token cap with thinking off; `cut` derives the 2,048-token view (the
note's budget) by cutting each completion at 2,048 tokens with the model's tokenizer. There is one run directory
per (model, prompt, draw), named <label>_<prompt>_<draw>, because `run_zeroshot.py grade` keeps one record per
(model, hint, task, idx). Samples are the 475 paired samples of results/per_sample.csv.

Config (environment, read by `generate`):
    CHAT_API_URL     required: an OpenAI-compatible chat-completions URL, e.g.
                     http://localhost:8000/v1/chat/completions for `vllm serve Qwen/Qwen3.6-27B`
    CHAT_API_KEY     required: the bearer token (any placeholder for a server without authentication)
    CHAT_API_EXTRA   optional JSON merged into every request body; default
                     {"chat_template_kwargs": {"enable_thinking": false}}, the switch the vLLM runs used.
                     `generate` reports any output that still contains thinking content.

Usage:
    python3 experiments/free/free_pilot.py generate --model Qwen/Qwen3.6-27B --label qwen3.6-27b \
        --hints layout,direct,free --draws greedy [--limit 9] [--workers 16]
    python3 experiments/free/free_pilot.py generate --model Qwen/Qwen3.6-27B --label qwen3.6-27b \
        --hints layout,direct,free --draws s1,s2,s3,s4,s5
    python3 experiments/free/free_pilot.py cut [--prefix qwen3.6-27b] [--tokenizer Qwen/Qwen3.6-27B]
    python3 experiments/free/free_pilot.py grade [--prefix qwen3.6-27b]
    python3 experiments/free/analyze_free.py
`cut` needs the `tokenizers` and `huggingface_hub` packages. All commands take --out (default runs/free_pilot).
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference/text"))
import run_zeroshot  # noqa: E402

SIZE, MAX_TOKENS, CUT = 512, 8192, 2048
FREE = "Think it through in whatever way you find useful, then give the ANSWER line."
run_zeroshot.HINTS["free"] = FREE
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0.0)
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
TOKENIZER = {"qwen3.5-4b": "Qwen/Qwen3.5-4B", "qwen3.5-9b": "Qwen/Qwen3.5-9B", "qwen3.6-27b": "Qwen/Qwen3.6-27B"}
BENCH = ROOT / "data/VBVR-Pro-Bench/VBVR-Pro-Bench-Video"
SPECS = ROOT / "inference/text/specs_v1.json"


def paired_samples():
    specs = {s["task"]: s for s in json.loads(SPECS.read_text())}
    rows = csv.DictReader((ROOT / "results/per_sample.csv").open())
    return [(specs[r["task"]], int(r["idx"])) for r in rows]


def decoding(draw):
    if draw == "greedy":
        return dict(temperature=0, seed=0)
    return dict(SAMPLING, seed=int(draw[1:]))


def endpoint():
    url, key = os.environ.get("CHAT_API_URL"), os.environ.get("CHAT_API_KEY")
    if not url or not key:
        sys.exit("CHAT_API_URL and CHAT_API_KEY must be set (see the module docstring)")
    extra = json.loads(os.environ["CHAT_API_EXTRA"]) if os.environ.get("CHAT_API_EXTRA") else THINKING_OFF
    print(f"endpoint {url}; extra request fields {sorted(extra)}", flush=True)
    return url, key, extra


def call(api, model, text, img_b64, draw, retries=20):
    url, key, extra = api
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": text},
        ]}],
        "max_tokens": MAX_TOKENS,
        **decoding(draw),
        **extra,
    }
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            if "choices" not in out:
                raise RuntimeError(f"no choices: {str(out)[:300]}")
            return out
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as e:
            code = getattr(e, "code", None)
            if code in (400, 401, 402, 403) or attempt == retries - 1:
                detail = e.read().decode()[:400] if hasattr(e, "read") else str(e)
                raise RuntimeError(f"API error {code}: {detail}") from e
            time.sleep(min(60, 2 ** attempt + 1) + random.uniform(0, 2))


def thinking(rec):
    details = (rec.get("usage") or {}).get("completion_tokens_details") or {}
    return "<think>" in rec["content"] or bool(rec.get("reasoning")) or bool(details.get("reasoning_tokens"))


def cmd_generate(args):
    api = endpoint()
    samples = paired_samples()
    if args.limit:
        samples = samples[:: max(1, len(samples) // args.limit)][: args.limit]
    stop = threading.Event()
    for draw in args.draws.split(","):
        for hint in args.hints.split(","):
            run = args.out / f"{args.label}_{hint}_{draw}"
            run.mkdir(parents=True, exist_ok=True)
            path = run / "responses.jsonl"
            done = set()
            if path.exists():
                done = {(r["task"], r["idx"]) for r in map(json.loads, path.read_text().splitlines()) if "error" not in r}
            todo = [(s, i) for s, i in samples if (s["task"], i) not in done]
            print(f"{run.name}: {len(done)} done, {len(todo)} to generate", flush=True)
            leaks = 0

            def one(spec, idx):
                if stop.is_set():
                    return None
                sample = BENCH / spec["split"] / spec["task"] / f"{idx:05d}"
                text, img = run_zeroshot.build_prompt(spec, sample, hint, size=SIZE)
                t0 = time.time()
                resp = call(api, args.model, text, img, draw)
                msg = resp["choices"][0]["message"]
                return dict(model=args.label, served_model=resp.get("model"), hint=hint, task=spec["task"],
                            split=spec["split"], idx=idx, input_px=SIZE, eval_size=SIZE, draw=draw,
                            decoding=decoding(draw), content=msg.get("content") or "",
                            reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
                            finish=resp["choices"][0].get("finish_reason"), usage=resp.get("usage"),
                            latency_s=round(time.time() - t0, 2))

            with ThreadPoolExecutor(args.workers) as ex, path.open("a") as f:
                futs = {ex.submit(one, s, i): (s, i) for s, i in todo}
                for n, fut in enumerate(as_completed(futs), 1):
                    spec, idx = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:  # recorded, not skipped silently
                        rec = dict(model=args.label, hint=hint, task=spec["task"], split=spec["split"], idx=idx,
                                   draw=draw, error=str(e)[:400])
                        if any(f"API error {c}" in rec["error"] for c in (401, 402, 403)):
                            stop.set()
                    if rec is None:
                        continue
                    leaks += "error" not in rec and thinking(rec)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if n % 25 == 0 or "error" in rec:
                        print(f"  {run.name} {n}/{len(todo)} {spec['task'][:28]} #{idx} "
                              f"{rec.get('finish', rec.get('error', ''))[:120]}", flush=True)
            print(f"{run.name}: finished this pass", flush=True)
            if leaks:
                print(f"WARNING {run.name}: {leaks} outputs contain thinking content; check CHAT_API_EXTRA", flush=True)
            if stop.is_set():
                sys.exit("stopped: the endpoint refused the key (401/402/403); rerun the same command to resume")


def cmd_cut(args):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    toks = {}
    for run in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name.startswith(args.prefix)
                      and not p.name.endswith(f"_cap{CUT}") and p.name != "logs"):
        recs = [r for r in map(json.loads, (run / "responses.jsonl").read_text().splitlines()) if "error" not in r]
        label = run.name.rsplit("_", 2)[0]
        repo = args.tokenizer or TOKENIZER.get(label)
        if not repo:
            sys.exit(f"no tokenizer known for {label}; pass --tokenizer <Hugging Face repo>")
        if repo not in toks:
            toks[repo] = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        tok, diffs, cut_n = toks[repo], [], 0
        dst = run.parent / f"{run.name}_cap{CUT}"
        dst.mkdir(exist_ok=True)
        with (dst / "responses.jsonl").open("w") as f:
            for r in recs:
                ids = tok.encode(r["content"], add_special_tokens=False).ids
                n = int(r["usage"]["completion_tokens"])
                diffs.append(len(ids) - n)
                if n > CUT:
                    cut_n += 1
                    r = dict(r, content=tok.decode(ids[:CUT]), finish="length",
                             usage=dict(r["usage"], completion_tokens=CUT))
                f.write(json.dumps(r) + "\n")
        off = sorted(set(diffs))
        print(f"{run.name}: {len(recs)} records, {cut_n} cut at {CUT}; tokenizer count - reported count "
              f"in {off[:6]}{'...' if len(off) > 6 else ''}")


def cmd_grade(args):
    for run in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name.startswith(args.prefix) and p.name != "logs"):
        subprocess.run([sys.executable, str(ROOT / "inference/text/run_zeroshot.py"), "grade", "--specs", str(SPECS),
                        "--bench", str(BENCH), "--out", str(run)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True, help="model name as the endpoint serves it")
    g.add_argument("--label", required=True, help="run-directory prefix, e.g. qwen3.6-27b")
    g.add_argument("--hints", default="layout,direct,free")
    g.add_argument("--draws", default="greedy", help="comma list of greedy and s<N>")
    g.add_argument("--limit", type=int, default=0, help="spread N samples over the task list (smoke test)")
    g.add_argument("--workers", type=int, default=16)
    for name in ("cut", "grade"):
        c = sub.add_parser(name)
        c.add_argument("--prefix", default="", help="only run directories whose name starts with this")
        if name == "cut":
            c.add_argument("--tokenizer", default="", help="Hugging Face repo of the tokenizer (default: by label)")
    for p in sub.choices.values():
        p.add_argument("--out", type=Path, default=ROOT / "runs/free_pilot")
    args = ap.parse_args()
    {"generate": cmd_generate, "cut": cmd_cut, "grade": cmd_grade}[args.cmd](args)


if __name__ == "__main__":
    main()
