#!/usr/bin/env python3
"""Generate one solution video per VBVR-Pro-Bench sample with a released VBVR-Pro Wan2.2 checkpoint.

A thin wrapper around the Wan2.2 Diffusers path of the official VBVR-Pro `example.py` (`run_wan_diffusers`): the same
pipeline classes, the repository's default negative prompt and seed handling, but the pipeline is loaded once and
reused for every sample. Settings of this study: 512 x 512, 81 frames, 50 denoising steps, classifier-free guidance
5, seed 42, 16 fps. Videos are written in the layout the VBVR-Pro-Bench evaluator expects:
<out>/<split>/<task>/<idx:05d>.mp4.

This wrapper was not the launcher that produced the released scores. The released G27 videos and G5's out-of-domain
videos used these settings (seed 42, the repository's negative prompt); G5's in-domain videos came from VBVR-Pro's
evaluation script (rl_training/src/cli/eval_i2v.py), whose negative prompt, per-sample seed and first-frame resize
differ. See the README.

Config (env): VBVR_PRO_DIR (required), a checkout of https://github.com/Video-Reason/VBVR-Pro at
f613643f01be6512ad07cdf28213da726e2c676e with its environment installed; `example.py` is imported from there.

Usage:
    VBVR_PRO_DIR=/path/to/VBVR-Pro python3 generate_videos.py --model Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B \
        --bench <VBVR-Pro-Bench-Video dir> --out <video dir> [--split In-Domain_50] [--tasks T1,T2] [--idx 0,1,2,3,4]
"""

import argparse
import os
import sys
from pathlib import Path


def samples(bench, split, tasks, idxs):
    splits = [split] if split else ["In-Domain_50", "Out-of-Domain_50"]
    for sp in splits:
        root = Path(bench) / sp
        if not root.is_dir():
            sys.exit(f"missing split directory {root}")
        for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if tasks and task_dir.name not in tasks:
                continue
            for idx in idxs:
                sample = task_dir / f"{idx:05d}"
                if sample.is_dir():
                    yield sp, task_dir.name, idx, sample


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="VBVR-Pro Wan2.2 checkpoint (Hugging Face id or local Diffusers dir)")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="", choices=["", "In-Domain_50", "Out-of-Domain_50"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--idx", default="0,1,2,3,4")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    pro = os.environ.get("VBVR_PRO_DIR")
    if not pro or not (Path(pro) / "example.py").is_file():
        sys.exit("VBVR_PRO_DIR must point at a VBVR-Pro checkout that contains example.py")
    sys.path.insert(0, pro)
    import example  # the official VBVR-Pro CLI module
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    todo = [s for s in samples(args.bench, args.split, set(args.tasks.split(",")) if args.tasks else None,
                               [int(i) for i in args.idx.split(",")])
            if not (Path(args.out) / s[0] / s[1] / f"{s[2]:05d}.mp4").exists()]
    print(f"{len(todo)} videos to generate with {args.model}", flush=True)
    if not todo:
        return
    example.require_cuda(torch, args.device)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to(args.device)
    for k, (split, task, idx, sample) in enumerate(todo, 1):
        example.set_seed(torch, args.seed)
        image = Image.open(sample / "first_frame.png").convert("RGB").resize((args.size, args.size))
        frames = pipe(prompt=(sample / "prompt.txt").read_text().strip(), negative_prompt=example.DEFAULT_NEGATIVE_PROMPT,
                      image=image, height=args.size, width=args.size, num_frames=args.num_frames,
                      num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                      generator=torch.Generator(device="cpu").manual_seed(args.seed)).frames[0]
        path = Path(args.out) / split / task / f"{idx:05d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(path), fps=args.fps)
        print(f"  {k}/{len(todo)} {path}", flush=True)


if __name__ == "__main__":
    main()
