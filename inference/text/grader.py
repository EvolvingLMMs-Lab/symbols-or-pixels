#!/usr/bin/env python3
"""Answer-level grading for the zero-shot language-model runs on VBVR-Pro-Bench.

A spec (see README.md in this directory) names the ANSWER key, the format shown to the model, and a
comparator. This module resolves metadata paths, grades an answer, builds the canonical GT answer, builds a
known-wrong answer, and runs the known-answer check (GT must pass, wrong must fail) on every sample.

Usage:
    python3 grader.py check --specs specs_v1.json --bench <VBVR-Pro-Bench-Video dir>
"""

import argparse
import json
import math
import re
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------- paths


def _parse(path):
    """'sgt.a[*].b[sgt.i].c[0]' -> [('key','sgt'),('key','a'),('all',),('key','b'),('ref','sgt.i'),...]"""
    steps, i, n = [], 0, len(path)
    while i < n:
        c = path[i]
        if c == ".":
            i += 1
        elif c == "[":
            depth, j = 1, i + 1
            while depth:
                depth += {"[": 1, "]": -1}.get(path[j], 0)
                j += 1
            inner = path[i + 1:j - 1].strip()
            if inner == "*":
                steps.append(("all",))
            elif re.fullmatch(r"-?\d+", inner):
                steps.append(("idx", int(inner)))
            else:
                steps.append(("ref", inner))
            i = j
        else:
            j = i
            while j < n and path[j] not in ".[":
                j += 1
            steps.append(("key", path[i:j]))
            i = j
    return steps


def _apply(value, steps, meta):
    for k, step in enumerate(steps):
        if step[0] == "all":
            return [_apply(v, steps[k + 1:], meta) for v in value]
        if step[0] == "key":
            value = value[step[1]]
        elif step[0] == "idx":
            value = value[step[1]]
        else:
            value = value[int(resolve(meta, step[1]))]
    return value


def resolve(meta, spec_value):
    """Resolve a path string, a {"path", "point_keys"} dict, or a literal (int/list) against metadata."""
    if isinstance(spec_value, dict) and "path" in spec_value:
        value = resolve(meta, spec_value["path"])
        keys = spec_value.get("point_keys")
        return _to_points(value, keys) if keys else value
    if not isinstance(spec_value, str):
        return spec_value
    steps = _parse(spec_value)
    root = steps[0][1]
    base = {"sgt": meta.get("semantic_ground_truth", {}), "par": meta.get("parameters", {}), "meta": meta}
    if root not in base:
        raise KeyError(f"path must start with sgt/par/meta: {spec_value}")
    return _apply(base[root], steps[1:], meta)


def _to_points(value, keys):
    if isinstance(value, dict):
        return [float(value[keys[0]]), float(value[keys[1]])]
    return [_to_points(v, keys) for v in value]


# ---------------------------------------------------------------- normalization


def norm_str(s):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]", " ", str(s).strip().lower()))


def norm(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float)):
        return round(float(x), 6)
    if isinstance(x, str):
        return norm_str(x)
    if isinstance(x, dict):
        return tuple(sorted((norm_str(k), norm(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(norm(v) for v in x)
    return x


def as_point(p):
    if isinstance(p, dict):
        for kx, ky in (("x", "y"), ("cx", "cy")):
            if kx in p and ky in p:
                return [float(p[kx]), float(p[ky])]
    return [float(p[0]), float(p[1])]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def to_int(x):
    if isinstance(x, bool):
        raise ValueError("bool is not an int answer")
    if isinstance(x, str):
        x = x.strip()
    f = float(x)
    if abs(f - round(f)) > 1e-6:
        raise ValueError(f"not an integer: {x}")
    return int(round(f))


# ---------------------------------------------------------------- grid / graph helpers


def _cells(cells, convention):
    return [tuple(c) if convention == "col_row" else (c[1], c[0]) for c in cells]


def _grid_env(p, meta):
    conv = p.get("convention", "col_row")
    wall = resolve(meta, p["wall_grid"]) if p.get("wall_grid") else None
    wall_value = p.get("wall_value", 1)
    if wall is not None:
        h, w = len(wall), len(wall[0])
    else:
        w, h = int(resolve(meta, p["grid_w"])), int(resolve(meta, p["grid_h"]))
    blocked = set(_cells(resolve(meta, p["blocked"]), conv)) if p.get("blocked") else set()
    if wall is not None:
        blocked |= {(c, r) for r in range(h) for c in range(w) if wall[r][c] == wall_value}
    start = _cells([resolve(meta, p["start"])], conv)[0]
    end = _cells([resolve(meta, p["end"])], conv)[0]
    waypoints = _cells(resolve(meta, p["waypoints"]), conv) if p.get("waypoints") else []
    cost = resolve(meta, p["cost_grid"]) if p.get("cost_grid") else None
    gt = _cells(resolve(meta, p["gt_path"]), conv)
    return dict(w=w, h=h, blocked=blocked, start=start, end=end, waypoints=waypoints, cost=cost, gt=gt)


def _bfs(env, a, b):
    seen, q = {a: 0}, deque([a])
    while q:
        c = q.popleft()
        if c == b:
            return seen[c]
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (c[0] + dc, c[1] + dr)
            if 0 <= nxt[0] < env["w"] and 0 <= nxt[1] < env["h"] and nxt not in env["blocked"] and nxt not in seen:
                seen[nxt] = seen[c] + 1
                q.append(nxt)
    return None


def _path_cost(env, path):
    return sum(env["cost"][r][c] for c, r in path)


def grid_path_check(p, meta, answer):
    env = _grid_env(p, meta)
    path = [tuple(int(v) for v in cell) for cell in answer]
    if len(path) < 1 or path[0] != env["start"] or path[-1] != env["end"]:
        return False, "wrong endpoints"
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False, f"non-adjacent step {a}->{b}"
    for c in path:
        if not (0 <= c[0] < env["w"] and 0 <= c[1] < env["h"]) or c in env["blocked"]:
            return False, f"blocked or out of bounds {c}"
    if p.get("objective", "shortest") == "max_cost":
        if len(set(path)) != len(path):
            return False, "revisits a cell"
        want = _path_cost(env, env["gt"])
        got = _path_cost(env, path)
        return (got == want), f"cost {got} vs optimum {want}"
    targets = env["waypoints"] + [env["end"]]
    pos, prev = 0, env["start"]
    for t in targets:
        try:
            nxt = path.index(t, pos)
        except ValueError:
            return False, f"misses waypoint {t}"
        need = _bfs(env, prev, t)
        if need is None or nxt - pos != need:
            return False, f"segment to {t} has {nxt - pos} steps, optimum {need}"
        pos, prev = nxt, t
    return pos == len(path) - 1, "ok" if pos == len(path) - 1 else "extra cells after end"


def graph_path_check(p, meta, answer):
    edges = {tuple(e) for e in resolve(meta, p["edges"])}
    start, end = resolve(meta, p["start"]), resolve(meta, p["end"])
    gt = resolve(meta, p["gt_path"])
    nodes = [to_int(v) for v in answer]
    if not nodes or nodes[0] != start or nodes[-1] != end:
        return False, "wrong endpoints"
    if any((a, b) not in edges for a, b in zip(nodes, nodes[1:])):
        return False, "uses a missing edge"
    return len(nodes) == len(gt), f"{len(nodes) - 1} edges vs optimum {len(gt) - 1}"


def stack_moves_check(p, meta, answer):
    stacks = [list(s) for s in resolve(meta, p["initial"])]
    target = [list(s) for s in resolve(meta, p["target"])]
    moves = [[to_int(a), to_int(b)] for a, b in answer]
    for a, b in moves:
        if not (0 <= a < len(stacks) and 0 <= b < len(stacks)) or not stacks[a] or a == b:
            return False, f"illegal move {a}->{b}"
        stacks[b].append(stacks[a].pop())
    optimal = int(resolve(meta, p["optimal"]))
    return stacks == target and len(moves) == optimal, f"{len(moves)} moves, reached={stacks == target}"


def sliding_moves_check(p, meta, answer):
    board = [list(r) for r in resolve(meta, p["initial"])]
    blank = p.get("blank_value", 0)
    h, w = len(board), len(board[0])
    r, c = next((i, j) for i in range(h) for j in range(w) if board[i][j] == blank)
    delta = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    moves = [norm_str(m) for m in answer]
    for m in moves:
        if m not in delta:
            return False, f"bad move {m}"
        nr, nc = r + delta[m][0], c + delta[m][1]
        if not (0 <= nr < h and 0 <= nc < w):
            return False, "moves off the board"
        board[r][c], board[nr][nc] = board[nr][nc], board[r][c]
        r, c = nr, nc
    goal = list(range(1, h * w)) + [blank]
    solved = [v for row in board for v in row] == goal
    need = int(resolve(meta, p["required_len"]))
    return solved and len(moves) == need, f"{len(moves)} moves, solved={solved}"


# ---------------------------------------------------------------- comparators


def _nearest(points, q):
    return min(range(len(points)), key=lambda i: dist(points[i], q))


def _target_index(p, meta, cands):
    if p.get("target_index") is not None:
        return int(resolve(meta, p["target_index"]))
    return _nearest(cands, as_point(resolve(meta, p["target"])))


def _target_set(p, meta, cands):
    if p.get("target_indices") is not None:
        return {int(i) for i in resolve(meta, p["target_indices"])}
    return {_nearest(cands, as_point(t)) for t in resolve(meta, p["target_points"])}


def _match_all(a, b, tol):
    """One-to-one matching with every pair within tol (small sets; greedy on sorted distances)."""
    if len(a) != len(b):
        return False
    pairs = sorted((dist(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b))
    used_a, used_b = set(), set()
    for d, i, j in pairs:
        if d <= tol and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return len(used_a) == len(a)


def grade_value(spec, meta, value):
    """Return (correct: bool, detail: str) for the answer value (already pulled out of the ANSWER object)."""
    p, t = spec["comparator"], spec["comparator"]["type"]
    if t == "int_exact":
        want = int(resolve(meta, p["gt"])) + int(p.get("offset", 0))
        got = to_int(value)
        return got == want, f"{got} vs {want}"
    if t == "str_exact":
        want = norm_str(resolve(meta, p["gt"]))
        got = norm_str(value)
        aliases = {norm_str(k): {norm_str(a) for a in v} for k, v in (p.get("aliases") or {}).items()}
        ok = got == want or got in aliases.get(want, set())
        return ok, f"{got!r} vs {want!r}"
    if t == "num_tol":
        want, got = float(resolve(meta, p["gt"])), float(value)
        return abs(got - want) <= float(p["tol"]), f"{got} vs {want}"
    if t == "rgb_tol":
        want, got = resolve(meta, p["gt"]), [float(v) for v in value]
        return len(got) == 3 and max(abs(a - b) for a, b in zip(got, want)) <= float(p["tol"]), f"{got} vs {want}"
    if t == "point_tol":
        want, got = as_point(resolve(meta, p["gt"])), as_point(value)
        return dist(got, want) <= float(p["tol"]), f"{dist(got, want):.1f}px"
    if t == "point_nearest":
        cands = [as_point(c) for c in resolve(meta, p["candidates"])]
        got = as_point(value)
        j = _nearest(cands, got)
        want = _target_index(p, meta, cands)
        return j == want and dist(cands[j], got) <= float(p["max_px"]), f"nearest {j} vs target {want}"
    if t == "point_set_nearest":
        cands = [as_point(c) for c in resolve(meta, p["candidates"])]
        want = _target_set(p, meta, cands)
        got = set()
        for q in value:
            q = as_point(q)
            j = _nearest(cands, q)
            if dist(cands[j], q) > float(p["max_px"]):
                return False, "a point is far from every candidate"
            got.add(j)
        return got == want and len(value) == len(want), f"{sorted(got)} vs {sorted(want)}"
    if t == "point_set_tol":
        want = [as_point(q) for q in resolve(meta, p["gt"])]
        got = [as_point(q) for q in value]
        return _match_all(got, want, float(p["tol"])), f"{len(got)} vs {len(want)} points"
    if t in ("list_exact", "set_exact"):
        gt = resolve(meta, p["gt"])
        if p.get("item"):
            gt = [g[p["item"]] for g in gt]
        a, b = [norm(v) for v in value], [norm(v) for v in gt]
        ok = a == b if t == "list_exact" else sorted(map(repr, a)) == sorted(map(repr, b))
        return ok, "match" if ok else "mismatch"
    if t == "fields_exact":
        for key, f in p["fields"].items():
            sub = dict(spec, comparator=dict(f, type={"exact": "list_exact", "str": "str_exact", "num_tol": "num_tol",
                                                      "rgb_tol": "rgb_tol", "list": "list_exact", "set": "set_exact",
                                                      "int": "int_exact", "point_tol": "point_tol"}[f.get("cmp", "exact")]))
            if key not in value:
                return False, f"missing field {key}"
            v = value[key]
            if sub["comparator"]["type"] == "list_exact" and not isinstance(v, list):
                ok = norm(v) == norm(resolve(meta, f["gt"]))
            else:
                ok, _ = grade_value(sub, meta, v)
            if not ok:
                return False, f"field {key} wrong"
        return True, "all fields"
    if t == "grid_path":
        return grid_path_check(p, meta, value)
    if t == "graph_path":
        return graph_path_check(p, meta, value)
    if t == "stack_moves":
        return stack_moves_check(p, meta, value)
    if t == "sliding_moves":
        return sliding_moves_check(p, meta, value)
    if t == "custom":
        from custom_graders import CUSTOM  # hand-written graders keyed by task name

        return CUSTOM[spec["task"]]["grade"](p, meta, value)
    raise ValueError(f"unknown comparator {t}")


def pull_value(spec, answer_obj):
    """Take the graded value out of a parsed ANSWER object."""
    if spec["comparator"]["type"] == "fields_exact":
        inner = answer_obj.get(spec["answer_key"]) if isinstance(answer_obj, dict) else None
        return inner if isinstance(inner, dict) else answer_obj
    if isinstance(answer_obj, dict):
        if spec["answer_key"] in answer_obj:
            return answer_obj[spec["answer_key"]]
        if len(answer_obj) == 1:
            return next(iter(answer_obj.values()))
        return answer_obj  # multi-field answer given flat instead of under answer_key
    return answer_obj


def grade(spec, meta, answer_obj):
    try:
        return grade_value(spec, meta, pull_value(spec, answer_obj))
    except Exception as e:  # malformed answers are wrong answers, with the reason kept
        return False, f"unparseable: {type(e).__name__}: {e}"


# ---------------------------------------------------------------- canonical and wrong answers


def canonical(spec, meta):
    p, t = spec["comparator"], spec["comparator"]["type"]
    if t == "int_exact":
        v = int(resolve(meta, p["gt"])) + int(p.get("offset", 0))
    elif t in ("str_exact", "num_tol"):
        v = resolve(meta, p["gt"])
    elif t == "rgb_tol":
        v = list(resolve(meta, p["gt"]))
    elif t == "point_tol":
        v = as_point(resolve(meta, p["gt"]))
    elif t == "point_nearest":
        cands = [as_point(c) for c in resolve(meta, p["candidates"])]
        v = cands[_target_index(p, meta, cands)]
    elif t == "point_set_nearest":
        cands = [as_point(c) for c in resolve(meta, p["candidates"])]
        v = [cands[i] for i in sorted(_target_set(p, meta, cands))]
    elif t == "point_set_tol":
        v = [as_point(q) for q in resolve(meta, p["gt"])]
    elif t in ("list_exact", "set_exact"):
        gt = resolve(meta, p["gt"])
        v = [g[p["item"]] for g in gt] if p.get("item") else list(gt)
    elif t == "fields_exact":
        v = {}
        for key, f in p["fields"].items():
            gt = resolve(meta, f["gt"])
            v[key] = int(gt) + int(f.get("offset", 0)) if f.get("cmp") == "int" else gt
        return {spec["answer_key"]: v}
    elif t == "grid_path":
        env = _grid_env(p, meta)
        v = [list(c) for c in env["gt"]]
    elif t in ("graph_path", "stack_moves", "sliding_moves"):
        v = resolve(meta, p["gt_path"] if t == "graph_path" else p["gt_moves"])
    elif t == "custom":
        from custom_graders import CUSTOM

        v = CUSTOM[spec["task"]]["canonical"](p, meta)
    else:
        raise ValueError(t)
    return {spec["answer_key"]: v}


def wrong(spec, meta):
    """A plausible wrong answer that the comparator must reject."""
    p, t = spec["comparator"], spec["comparator"]["type"]
    good = canonical(spec, meta)[spec["answer_key"]]
    if t == "int_exact":
        v = good + 1
    elif t == "str_exact":
        v = "definitely wrong"
    elif t == "num_tol":
        v = float(good) + 10 * float(p["tol"]) + 1
    elif t == "rgb_tol":
        v = [(c + 128) % 256 for c in good]
    elif t == "point_tol":
        v = [good[0] + 5 * float(p["tol"]) + 50, good[1]]
    elif t == "point_nearest":
        cands = [as_point(c) for c in resolve(meta, p["candidates"])]
        want = _target_index(p, meta, cands)
        v = cands[max((i for i in range(len(cands)) if i != want), key=lambda i: dist(cands[i], cands[want]))]
    elif t == "point_set_nearest":
        cands = [as_point(c) for c in resolve(meta, p["candidates"])]
        want = _target_set(p, meta, cands)
        others = [cands[i] for i in range(len(cands)) if i not in want]
        v = good[:-1] if len(good) > 1 or not others else good + others[:1]
        if v == good:
            v = good + others[:1]
    elif t == "point_set_tol":
        v = [[good[0][0] + 5 * float(p["tol"]) + 50, good[0][1]]] + good[1:]
    elif t == "list_exact":
        v = list(reversed(good)) if len(good) > 1 and list(reversed(good)) != good else good + [good[0]]
    elif t == "set_exact":
        v = good[:-1] if good else ["x"]
    elif t == "fields_exact":
        first = next(iter(p["fields"]))
        g = dict(good)
        g[first] = "definitely wrong" if not isinstance(g[first], (int, float)) else g[first] + 1000
        return {spec["answer_key"]: g}
    elif t in ("grid_path", "graph_path"):
        v = good[:1] + good[2:] if len(good) > 2 else good[:-1]
    elif t in ("stack_moves", "sliding_moves"):
        v = good[:-1]
    elif t == "custom":
        from custom_graders import CUSTOM

        v = CUSTOM[spec["task"]]["wrong"](p, meta)
    else:
        raise ValueError(t)
    return {spec["answer_key"]: v}


# ---------------------------------------------------------------- known-answer check


def load_meta(bench, spec, idx):
    return json.loads((Path(bench) / spec["split"] / spec["task"] / f"{idx:05d}" / "metadata.json").read_text())


def check_specs(specs, bench):
    report = []
    for spec in specs:
        if not spec.get("include", True):
            report.append((spec["task"], "excluded", spec.get("exclude_reason")))
            continue
        fails = []
        for idx in range(5):
            meta = load_meta(bench, spec, idx)
            try:
                ok_gt, why_gt = grade(spec, meta, canonical(spec, meta))
                ok_bad, why_bad = grade(spec, meta, wrong(spec, meta))
            except Exception as e:
                fails.append(f"{idx}: {type(e).__name__}: {e}")
                continue
            if not ok_gt:
                fails.append(f"{idx}: GT rejected ({why_gt})")
            if ok_bad:
                fails.append(f"{idx}: wrong answer accepted ({why_bad})")
        report.append((spec["task"], "PASS" if not fails else "FAIL", "; ".join(fails)))
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--specs", required=True)
    c.add_argument("--bench", required=True)
    args = ap.parse_args()
    specs = json.loads(Path(args.specs).read_text())
    rows = check_specs(specs, args.bench)
    for task, status, why in rows:
        print(f"{status:8s} {task} {why or ''}")
    n_pass = sum(r[1] == "PASS" for r in rows)
    n_excl = sum(r[1] == "excluded" for r in rows)
    print(f"\n{n_pass} pass, {sum(r[1] == 'FAIL' for r in rows)} fail, {n_excl} excluded, of {len(rows)}")
    raise SystemExit(0 if all(r[1] != "FAIL" for r in rows) else 1)


if __name__ == "__main__":
    main()
