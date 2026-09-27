#!/usr/bin/env python3
"""Static versions of Figures 1-4 of the note, drawn from results/summary/summary.json.

    fig1_cost.png      solve rate against forward FLOPs per answer, 95% CI whiskers with end caps
    fig2_groups.png    paired difference (text - video, pp) by split, task class and answer kind, CIs with end caps
    fig3_outcomes.png  paired samples by outcome (both / text only / video only / neither) per group
    fig4_failures.png  text answer outcomes per model and prompt; v2 score histograms of the two video models

Figures 2 and 3 show one configuration (pair, prompt, video threshold); the default is the note's primary one.

Usage:
    python3 analysis/plot.py [--root .] [--pair 27B] [--prompt layout] [--cut 0.9] [--format png]
"""

import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT, VIDEO, NEUTRAL = "#03639a", "#c2410c", "#6b7280"
BOTH, NEITHER = "#374151", "#e5e7eb"
OUTCOME_COLORS = {"correct": TEXT, "wrong": "#9ca3af", "no_answer": "#d1d5db", "truncated": "#f3f4f6"}
CAP = 4

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def pp(x):
    # Halves round away from zero, as JavaScript toFixed does in the note's figures (-6.25 -> -6.3).
    return f"{Decimal(x).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):+.1f}".replace("-", "\u2212")


def fig_cost(S, cut, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    key = f"solve_{cut}"
    pts = {}
    for name, e in S["systems"].items():
        x, r = e["flops"], e[key]
        y, lo, hi = 100 * r["rate"], 100 * r["ci"][0], 100 * r["ci"][1]
        pts[name] = (x, y)
        if e["kind"] == "video":
            style = dict(marker="s", color=VIDEO, mfc=VIDEO)
        elif name.endswith("layout"):
            style = dict(marker="o", color=TEXT, mfc=TEXT)
        else:
            style = dict(marker="o", color=TEXT, mfc="white")
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], capsize=CAP, lw=1.2, ms=6, ls="none", **style)
        if not name.endswith("direct"):  # direct-prompt points sit next to their layout point
            ax.annotate(e["label"].split(" (")[0].split(",")[0], (x, y), xytext=(6, -10),
                        textcoords="offset points", fontsize=7, color="#374151")
    for e in S["table3"]:
        if e["prompt"] != "layout":
            continue
        (xv, yv), (xt, yt) = pts[e["video"]], pts[e["text"]]
        ax.plot([xt, xv], [yt, yv], ls="--", lw=0.8, color=NEUTRAL, zorder=0)
        ax.annotate(f"{e['pair']}: {e['flops_ratio']:,.0f}\u00d7 fewer FLOPs", ((xt * xv) ** 0.5, (yt + yv) / 2),
                    xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7, color=NEUTRAL)
    ax.legend(handles=[plt.Line2D([], [], marker="s", color=VIDEO, ls="none", label="Video, VBVR-Pro fine-tuned"),
                       plt.Line2D([], [], marker="o", color=TEXT, ls="none", label="Text, layout prompt"),
                       plt.Line2D([], [], marker="o", color=TEXT, mfc="white", ls="none", label="Text, direct prompt")],
              fontsize=7, frameon=False, loc="upper right")
    ax.set_xscale("log")
    ax.set_xlabel("Forward FLOPs per answer (log scale)")
    ax.set_ylabel(f"Solve rate (%), video v2 \u2265 {cut}")
    ax.set_title("Figure 1: solve rate against forward FLOPs per answer", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def config_groups(S, pair, prompt, cut):
    rows = [g for g in S["groups"] if g["pair"] == pair and g["prompt"] == prompt and g["cut"] == cut]
    if not rows:
        sys.exit(f"no groups for pair={pair} prompt={prompt} cut={cut}")
    return rows


def fig_groups(S, pair, prompt, cut, path):
    rows = config_groups(S, pair, prompt, cut)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels, y, block = [], 0, None
    ticks = []
    for g in rows:
        if g["block"] != block:
            block = g["block"]
            y -= 0.6
            ax.text(-0.02, y, block.upper(), transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7, color=TEXT, weight="bold")
            y -= 0.8
        d, lo, hi = 100 * g["diff"], 100 * g["ci"][0], 100 * g["ci"][1]
        color = TEXT if lo > 0 else VIDEO if hi < 0 else NEUTRAL
        ax.errorbar([d], [y], xerr=[[d - lo], [hi - d]], fmt="o", color=color, capsize=CAP, lw=1.4, ms=5)
        ax.text(1.01, y, f"{pp(d)} [{pp(lo)}, {pp(hi)}]", transform=ax.get_yaxis_transform(), va="center",
                fontsize=7)
        ticks.append(y)
        labels.append(f"{g['label']} ({g['tasks']} tasks)")
        y -= 1
    ax.axvline(0, color="black", lw=0.8)
    ax.set_ylim(y + 0.5, 0)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Solve-rate difference, text \u2212 video (pp); left favors video, right favors text")
    ax.set_title(f"Figure 2: {pair} pair, {prompt} prompt, video v2 \u2265 {cut}", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_outcomes(S, pair, prompt, cut, path):
    rows = config_groups(S, pair, prompt, cut)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    parts = [("both", "Both solve", BOTH), ("text_only", "Text only", TEXT), ("video_only", "Video only", VIDEO),
             ("neither", "Neither", NEITHER)]
    for i, g in enumerate(rows):
        total, left = sum(g["cells"].values()), 0
        for k, label, color in parts:
            w = 100 * g["cells"][k] / total
            ax.barh(i, w, left=left, color=color, label=label if i == 0 else None, edgecolor="white", lw=0.5)
            left += w
        best = 100 * max(g["text"]["rate"], g["video"]["rate"])
        ax.text(101, i, f"union {100 * g['union']['rate']:.1f}% | best single {best:.1f}%", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [f"{g['label']} ({g['tasks']})" for g in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of paired samples (%)")
    ax.legend(ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title(f"Figure 3: {pair} pair, {prompt} prompt, video v2 \u2265 {cut}", loc="left", fontsize=9, pad=18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_failures(S, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    names = list(S["text_outcomes"])
    for i, name in enumerate(names):
        counts, left = S["text_outcomes"][name]["overall"], 0
        total = sum(counts.values())
        for k in ("correct", "wrong", "no_answer", "truncated"):
            w = 100 * counts[k] / total
            ax1.barh(i, w, left=left, color=OUTCOME_COLORS[k], edgecolor="#9ca3af", lw=0.4,
                     hatch="///" if k == "truncated" else None,
                     label={"no_answer": "Unparsable", "truncated": "Truncated at 2,048 tokens"}.get(k, k.title())
                     if i == 0 else None)
            left += w
    ax1.set_yticks(range(len(names)), [S["systems"][n]["label"].replace(" prompt", "") for n in names], fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of the 475 paired samples (%)")
    ax1.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    edges = [i / 10 for i in range(10)]
    for k, (name, color) in enumerate([("g5", "#f59e0b"), ("g27", VIDEO)]):
        hist = S["video_scores"][name]["hist"]
        ax2.bar([e + 0.02 + 0.03 * k for e in edges], hist, width=0.03, align="edge", color=color,
                label=S["systems"][name]["label"].split(" (")[0])
    ax2.axvspan(0.7, 0.9, color=NEUTRAL, alpha=0.12, lw=0)
    ax2.text(0.8, ax2.get_ylim()[1] * 0.95, "near miss\n0.7\u20130.9", ha="center", va="top", fontsize=6, color=NEUTRAL)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("v2 score (bins of 0.1)")
    ax2.set_ylabel("Videos")
    ax2.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Figure 4: text answer outcomes (left) and video v2 scores (right)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="repository root")
    ap.add_argument("--pair", default="27B", choices=["5B", "27B"])
    ap.add_argument("--prompt", default="layout", choices=["layout", "direct"])
    ap.add_argument("--cut", default="0.9", choices=["0.9", "0.7"])
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()
    root = Path(args.root)
    path = root / "results/summary/summary.json"
    if not path.exists():
        sys.exit(f"missing {path}; run analysis/analyze.py first")
    S = json.loads(path.read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig_cost(S, args.cut, out / f"fig1_cost.{args.format}")
    fig_groups(S, args.pair, args.prompt, args.cut, out / f"fig2_groups.{args.format}")
    fig_outcomes(S, args.pair, args.prompt, args.cut, out / f"fig3_outcomes.{args.format}")
    fig_failures(S, out / f"fig4_failures.{args.format}")
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
