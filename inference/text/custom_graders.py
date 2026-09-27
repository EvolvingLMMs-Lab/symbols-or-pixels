"""Hand-written graders for specs whose comparator type is `custom`.

CUSTOM[task] = {"grade": f(p, meta, value) -> (bool, str), "canonical": f(p, meta) -> value,
"wrong": f(p, meta) -> value}, where p is the spec's comparator dict. grader.py dispatches here, and
the known-answer check covers these graders too.
"""

from grader import as_point, dist, resolve, to_int


def _nearest(points, q):
    return min(range(len(points)), key=lambda i: dist(points[i], q))


# ---- ordered list of current object centres (G-174: circles sorted by circumference)


def ordered_points_grade(p, meta, value):
    cands = [as_point(c) for c in resolve(meta, p["candidates"])]
    ids = resolve(meta, p["ids"])
    want = list(resolve(meta, p["gt_order"]))
    if len(value) != len(cands):
        return False, f"{len(value)} points vs {len(cands)} objects"
    mapped = []
    for q in value:
        q = as_point(q)
        j = _nearest(cands, q)
        if dist(cands[j], q) > float(p["max_px"]):
            return False, "a point is far from every object"
        mapped.append(j)
    if len(set(mapped)) != len(mapped):
        return False, "an object is listed twice"
    got = [ids[j] for j in mapped]
    return got == want, f"{got} vs {want}"


def ordered_points_canonical(p, meta):
    cands = [as_point(c) for c in resolve(meta, p["candidates"])]
    ids = list(resolve(meta, p["ids"]))
    return [cands[ids.index(g)] for g in resolve(meta, p["gt_order"])]


def ordered_points_wrong(p, meta):
    return list(reversed(ordered_points_canonical(p, meta)))


# ---- matched from/to moves (G-24: each object slides to its own dashed outline)


def _moves_ends(p, meta):
    starts = [as_point(c) for c in resolve(meta, p.get("starts", p.get("from_points")))]
    ends = [as_point(c) for c in resolve(meta, p.get("ends", p.get("to_points")))]
    return starts, ends


def matched_moves_grade(p, meta, value):
    starts, ends = _moves_ends(p, meta)
    if len(value) != len(starts):
        return False, f"{len(value)} moves vs {len(starts)} objects"
    used = set()
    for move in value:
        a, b = as_point(move["from"]), as_point(move["to"])
        j = _nearest(starts, a)
        if dist(starts[j], a) > float(p["from_max_px"]) or j in used:
            return False, "a 'from' point matches no unused object"
        used.add(j)
        if dist(ends[j], b) > float(p["to_tol"]):
            return False, f"object {j} ends {dist(ends[j], b):.0f}px from its target"
    return True, "all moves"


def matched_moves_canonical(p, meta):
    starts, ends = _moves_ends(p, meta)
    return [{"from": s, "to": e} for s, e in zip(starts, ends)]


def matched_moves_wrong(p, meta):
    moves = matched_moves_canonical(p, meta)
    if len(moves) > 1:
        moves[0]["to"], moves[1]["to"] = moves[1]["to"], moves[0]["to"]
    else:
        moves[0]["to"] = [moves[0]["to"][0] + 200, moves[0]["to"][1]]
    return moves



# ---- point inside the target polygon (G-138: shapes are stored only as polygons)


def _in_poly(pt, poly):
    x, y, inside = pt[0], pt[1], False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def _seg_dist(q, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return dist(q, [a[0] + t * dx, a[1] + t * dy])


def _poly_dist(q, poly):
    return 0.0 if _in_poly(q, poly) else min(_seg_dist(q, a, b) for a, b in zip(poly, poly[1:] + poly[:1]))


def _polys(p, meta):
    return [[as_point(v) for v in poly] for poly in resolve(meta, p["polygons"])]


def polygon_grade(p, meta, value):
    polys, tgt = _polys(p, meta), int(resolve(meta, p["target_index"]))
    q = as_point(value)
    d = [_poly_dist(q, poly) for poly in polys]
    k = min(range(len(d)), key=d.__getitem__)
    return k == tgt and d[tgt] <= float(p["max_px"]), f"nearest polygon {k} vs {tgt}, {d[tgt]:.0f}px"


def _vertex_mean(poly):
    return [sum(v[0] for v in poly) / len(poly), sum(v[1] for v in poly) / len(poly)]


def polygon_canonical(p, meta):
    return _vertex_mean(_polys(p, meta)[int(resolve(meta, p["target_index"]))])


def polygon_wrong(p, meta):
    polys, tgt = _polys(p, meta), int(resolve(meta, p["target_index"]))
    c = _vertex_mean(polys[tgt])
    return min((_vertex_mean(q) for k, q in enumerate(polys) if k != tgt), key=lambda m: dist(m, c))


# ---- sorted line (G-3: types contiguous, sizes strictly increasing inside a type, either group first)


def sorted_line_grade(p, meta, value):
    centers = [as_point(c) for c in resolve(meta, p["centers"])]
    types, sizes = resolve(meta, p["types"]), resolve(meta, p["sizes"])
    if len(value) != len(centers):
        return False, f"{len(value)} points vs {len(centers)} objects"
    ks = []
    for q in value:
        q = as_point(q)
        j = _nearest(centers, q)
        if dist(centers[j], q) > float(p["max_px"]):
            return False, "a point is far from every object"
        ks.append(j)
    if sorted(ks) != list(range(len(centers))):
        return False, "objects missing or listed twice"
    seq = [types[k] for k in ks]
    runs = [t for i, t in enumerate(seq) if i == 0 or seq[i - 1] != t]
    if len(runs) != len(set(seq)):
        return False, "a type group is split"
    ok = all(sizes[a] < sizes[b] for a, b in zip(ks, ks[1:]) if types[a] == types[b])
    return ok, "sorted" if ok else "sizes not increasing inside a group"


def sorted_line_canonical(p, meta):
    centers = [as_point(c) for c in resolve(meta, p["centers"])]
    ids = list(resolve(meta, p["ids"]))
    return [centers[ids.index(i)] for i in resolve(meta, p["gt_order"])]


def sorted_line_wrong(p, meta):
    return list(reversed(sorted_line_canonical(p, meta)))


# ---- grid path through one waypoint with a two-part GT path (G-45: agent -> key -> door)


def _via_waypoint_params(p, meta):
    parts = [resolve(meta, q) for q in p["gt_path_parts"]]
    gt = list(parts[0])
    for part in parts[1:]:
        gt += list(part[1:])
    return dict(p, waypoints=[resolve(meta, w) for w in p["waypoints"]], gt_path=gt)


def via_waypoint_grade(p, meta, value):
    from grader import grid_path_check

    return grid_path_check(_via_waypoint_params(p, meta), meta, value)


def via_waypoint_canonical(p, meta):
    from grader import _grid_env

    return [list(c) for c in _grid_env(_via_waypoint_params(p, meta), meta)["gt"]]


def via_waypoint_wrong(p, meta):
    from grader import _bfs, _grid_env

    env = _grid_env(_via_waypoint_params(p, meta), meta)
    gt = [tuple(c) for c in env["gt"]]
    # the shortest start -> end route that skips the key, else a path that stops one cell early
    prev, route = {env["start"]: None}, [env["start"]]
    from collections import deque

    q = deque([env["start"]])
    while q:
        c = q.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dc, c[1] + dr)
            if 0 <= n[0] < env["w"] and 0 <= n[1] < env["h"] and n not in env["blocked"] and n not in prev:
                prev[n] = c
                q.append(n)
    if env["end"] in prev:
        route, c = [], env["end"]
        while c is not None:
            route.append(list(c))
            c = prev[c]
        route.reverse()
        if tuple(env["waypoints"][0]) not in {tuple(r) for r in route}:
            return route
    return [list(c) for c in gt[:-1]]


# ---- point set with the target set given by a boolean mask (G-9)


def _masked(p, meta):
    cands = [as_point(c) for c in resolve(meta, p["candidates"])]
    return cands, {k for k, t in enumerate(resolve(meta, p["target_mask"])) if t}


def masked_set_grade(p, meta, value):
    cands, want = _masked(p, meta)
    got = []
    for q in value:
        q = as_point(q)
        j = _nearest(cands, q)
        if dist(cands[j], q) > float(p["max_px"]):
            return False, "a point is far from every object"
        got.append(j)
    ok = len(set(got)) == len(got) and set(got) == want
    return ok, f"{sorted(got)} vs {sorted(want)}"


def masked_set_canonical(p, meta):
    cands, want = _masked(p, meta)
    return [cands[k] for k in sorted(want)]


def masked_set_wrong(p, meta):
    cands, want = _masked(p, meta)
    good = [cands[k] for k in sorted(want)]
    others = [cands[k] for k in range(len(cands)) if k not in want]
    return good[:-1] if len(good) > 1 else good + others[:1]


# ---- O-14: size ratio and style change of the "?" shape


def _o14_truth(p, meta):
    ratio = float(resolve(meta, p["size_ratio_numerator"])) / float(resolve(meta, p["size_ratio_denominator"]))
    f, t, rank = resolve(meta, p["from_style"]), resolve(meta, p["to_style"]), p["outline_rank"]
    if t == "filled":
        s = "filled"
    elif f == "filled":
        s = "outline"
    elif rank[t] > rank[f]:
        s = "thicker_outline"
    else:
        s = "thinner_outline"
    return ratio, s


def o14_grade(p, meta, value):
    from grader import norm_str

    ratio, s = _o14_truth(p, meta)
    ok_r = abs(float(value["size_ratio"]) - ratio) <= float(p["size_tol"])
    ok_s = norm_str(value["style_change"]) == norm_str(s)
    return ok_r and ok_s, f"ratio {value['size_ratio']} vs {ratio:.3f}, style {value['style_change']} vs {s}"


def o14_canonical(p, meta):
    ratio, s = _o14_truth(p, meta)
    return {"size_ratio": round(ratio, 3), "style_change": s}


def o14_wrong(p, meta):
    ratio, s = _o14_truth(p, meta)
    return {"size_ratio": round(ratio, 3), "style_change": "outline" if s == "filled" else "filled"}


# ---- O-29: absorb order, valid when every absorbed cluster is strictly smaller than the red one so far


def _o29(p, meta):
    letters = [str(v).upper() for v in resolve(meta, p["letters"])]
    return letters, resolve(meta, p["sizes"]), resolve(meta, p["is_red"]).index(True)


def o29_grade(p, meta, value):
    letters, sizes, red = _o29(p, meta)
    got = [str(v).strip().upper() for v in value]
    if sorted(got) != sorted(letters[i] for i in range(len(letters)) if i != red):
        return False, "not a permutation of the other clusters"
    size = sizes[red]
    for a in got:
        s = sizes[letters.index(a)]
        if not s < size:
            return False, f"{a} ({s}) is not smaller than {size}"
        size += s
    return True, "valid order"


def o29_canonical(p, meta):
    return [str(v).upper() for v in resolve(meta, p["reference_order"])]


def o29_wrong(p, meta):
    rev = list(reversed(o29_canonical(p, meta)))
    return rev if not o29_grade(p, meta, rev)[0] else rev[:-1]


# ---- O-30: gap per waiting book; books of equal height (within height_tol) are interchangeable


def _o30(p, meta):
    pos = {int(k): v for k, v in resolve(meta, p["insertion_indices"]).items()}
    rank = {r: 1 + sum(pos[r2] < pos[r] for r2 in pos) for r in pos}
    return resolve(meta, p["heights"]), resolve(meta, p["queue_order"]), rank, {v: k for k, v in rank.items()}


def o30_grade(p, meta, value):
    heights, queue, rank, by_rank = _o30(p, meta)
    got = [to_int(v) for v in value]
    if len(got) != len(queue) or sorted(got) != list(range(1, len(queue) + 1)):
        return False, "not one gap per book"
    ok = all(abs(heights[queue[k]] - heights[by_rank[g]]) <= float(p["height_tol"]) for k, g in enumerate(got))
    return ok, f"{got} vs {[rank[q] for q in queue]}"


def o30_canonical(p, meta):
    _, queue, rank, _ = _o30(p, meta)
    return [rank[q] for q in queue]


def o30_wrong(p, meta):
    good = o30_canonical(p, meta)
    rev = list(reversed(good))
    return rev if not o30_grade(p, meta, rev)[0] else good + [len(good) + 1]


# ---- O-31: eat order of prey points, valid when each prey is strictly smaller than the black ball so far


def _o31(p, meta):
    centers = [as_point(c) for c in resolve(meta, p["prey_centers"])]
    return centers, resolve(meta, p["prey_sizes"]), float(resolve(meta, p["black_size"])), float(resolve(meta, p["growth_factor"]))


def o31_grade(p, meta, value):
    centers, sizes, d, g = _o31(p, meta)
    if len(value) != len(centers):
        return False, f"{len(value)} points vs {len(centers)} prey"
    seen = []
    for q in value:
        q = as_point(q)
        k = _nearest(centers, q)
        if dist(centers[k], q) > sizes[k] / 2 + float(p["match_margin_px"]) or k in seen:
            return False, "a point matches no unused prey"
        seen.append(k)
        if not sizes[k] < d:
            return False, f"prey {k} ({sizes[k]}) is not smaller than {d:.1f}"
        d *= g
    return True, "valid order"


def o31_canonical(p, meta):
    centers = _o31(p, meta)[0]
    return [centers[int(i)] for i in resolve(meta, p["reference_order"])]


def o31_wrong(p, meta):
    rev = list(reversed(o31_canonical(p, meta)))
    return rev if not o31_grade(p, meta, rev)[0] else rev[:-1]


# ---- O-32: resting ball position, taken from the declarative render (sgt holds only 3D coordinates)


def _o32_gt(p, meta):
    layers = meta
    for key in p["gt_root"].split("."):
        layers = layers[key]
    hits = [l for l in layers if all(l.get(k) == v for k, v in p["gt_layer_filter"].items())]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} layers match the ball filter")
    kx, ky = p["gt_point_keys"]
    return [float(hits[0][kx]), float(hits[0][ky])]


def o32_grade(p, meta, value):
    d = dist(as_point(value), _o32_gt(p, meta))
    return d <= float(p["tol"]), f"{d:.1f}px"


def o32_canonical(p, meta):
    return _o32_gt(p, meta)


def o32_wrong(p, meta):
    gt = _o32_gt(p, meta)
    return [gt[0] + 5 * float(p["tol"]) + 50, gt[1]]


# ---- O-34: dots in connection order (objects are stored in number order)


def o34_grade(p, meta, value):
    pts = [as_point(c) for c in resolve(meta, p["points"])]
    nums = list(resolve(meta, p["numbers"]))
    if nums != list(range(1, len(pts) + 1)):
        raise ValueError("objects are not stored in number order")
    if len(value) != len(pts):
        return False, f"{len(value)} points vs {len(pts)} dots"
    ok = all(dist(as_point(a), b) <= float(p["max_px"]) for a, b in zip(value, pts))
    return ok, "in order" if ok else "a dot is out of order"


def o34_canonical(p, meta):
    return [as_point(c) for c in resolve(meta, p["points"])]


def o34_wrong(p, meta):
    return list(reversed(o34_canonical(p, meta)))


# ---- O-45: position of an earlier element identical to the missing one


def _o45(p, meta):
    return resolve(meta, p["values"]), int(resolve(meta, p["answer_index"])), resolve(meta, p["answer_value"])


def o45_grade(p, meta, value):
    import json

    vals, a, v = _o45(p, meta)
    k = to_int(value)
    ok = 1 <= k <= a and json.dumps(vals[k - 1]) == json.dumps(v)
    return ok, f"position {k}"


def o45_canonical(p, meta):
    import json

    vals, a, v = _o45(p, meta)
    return next(k + 1 for k in range(a) if json.dumps(vals[k]) == json.dumps(v))


def o45_wrong(p, meta):
    import json

    vals, a, v = _o45(p, meta)
    return next(k + 1 for k in range(a) if json.dumps(vals[k]) != json.dumps(v))


# ---- G-250: bounding box of the region inside all three circles, per-edge tolerance


def _bbox(p, meta):
    poly = [as_point(v) for v in resolve(meta, p["region_polygon"])]
    return [min(v[0] for v in poly), min(v[1] for v in poly), max(v[0] for v in poly), max(v[1] for v in poly)]


def bbox_grade(p, meta, value):
    gt, got = _bbox(p, meta), [float(v) for v in value]
    if len(got) != 4 or not (got[0] < got[2] and got[1] < got[3]):
        return False, "not a box"
    err = max(abs(a - b) for a, b in zip(got, gt))
    return err <= float(p["tol"]), f"max edge error {err:.0f}px"


def bbox_canonical(p, meta):
    return _bbox(p, meta)


def bbox_wrong(p, meta):
    x0, y0, x1, y1 = _bbox(p, meta)
    s = 3 * float(p["tol"])
    return [x0 - s, y0 - s, x1 + s, y1 + s]


# ---- G-54: unordered same-colour pairs of shape centres covering every shape


def _pairs(p, meta):
    return [as_point(c) for c in resolve(meta, p["candidates"])], resolve(meta, p["candidate_colors"]), int(resolve(meta, p["num_pairs"]))


def pairs_grade(p, meta, value):
    cands, cols, n = _pairs(p, meta)
    if len(value) != n:
        return False, f"{len(value)} pairs vs {n}"
    used = []
    for pair in value:
        if len(pair) != 2:
            return False, "a pair does not have two points"
        ij = []
        for q in pair:
            q = as_point(q)
            j = _nearest(cands, q)
            if dist(cands[j], q) > float(p["max_px"]):
                return False, "a point is far from every shape"
            ij.append(j)
        if ij[0] == ij[1] or cols[ij[0]] != cols[ij[1]]:
            return False, "a pair joins different colours"
        used += ij
    ok = len(set(used)) == len(used) == len(cands)
    return ok, "all pairs" if ok else "a shape is missing or used twice"


def pairs_canonical(p, meta):
    cands, cols, _ = _pairs(p, meta)
    groups = {}
    for c, col in zip(cands, cols):
        groups.setdefault(tuple(col), []).append(c)
    return [g for g in groups.values()]


def pairs_wrong(p, meta):
    good = pairs_canonical(p, meta)
    if len(good) > 1:
        good[0][1], good[1][1] = good[1][1], good[0][1]
    return good


# ---- O-56: an earlier visible cell identical to the missing Raven cell


def _cells(p, meta):
    miss = resolve(meta, p["missing_cell"])
    ans = resolve(meta, p["answer_cell"])
    same = lambda o: all(o[k] == ans[k] for k in ("shapes", "positions", "colors"))
    visible = [o for o in resolve(meta, p["cells"]) if (o["row"], o["col"]) != (miss["row"], miss["col"])]
    return visible, same


def raven_grade(p, meta, value):
    r, c = (to_int(v) for v in value)
    visible, same = _cells(p, meta)
    hit = [o for o in visible if (o["row"], o["col"]) == (r, c)]
    ok = len(hit) == 1 and same(hit[0])
    return ok, f"cell {[r, c]}"


def raven_canonical(p, meta):
    visible, same = _cells(p, meta)
    o = next(o for o in visible if same(o))
    return [o["row"], o["col"]]


def raven_wrong(p, meta):
    visible, same = _cells(p, meta)
    o = next(o for o in visible if not same(o))
    return [o["row"], o["col"]]


# ---- G-47: collect every key in any order, then reach the door, with the shortest total length


def _keys_env(p, meta):
    from grader import _cells as cells_of
    from grader import _grid_env

    env = _grid_env(dict(p, waypoints=None, gt_path=[]), meta)
    keys = resolve(meta, p["keys"]) if isinstance(p["keys"], str) else [resolve(meta, k) for k in p["keys"]]
    env["keys"] = cells_of(keys, p.get("convention", "col_row"))
    return env


def _optimum(env):
    import itertools

    from grader import _bfs

    best = None
    for order in itertools.permutations(env["keys"]):
        stops, total = [env["start"], *order, env["end"]], 0
        for a, b in zip(stops, stops[1:]):
            d = _bfs(env, a, b)
            if d is None:
                total = None
                break
            total += d
        if total is not None and (best is None or total < best):
            best = total
    return best


def keys_grade(p, meta, value):
    env = _keys_env(p, meta)
    path = [tuple(int(v) for v in cell) for cell in value]
    if not path or path[0] != env["start"] or path[-1] != env["end"]:
        return False, "wrong endpoints"
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False, f"non-adjacent step {a}->{b}"
    if any(not (0 <= c[0] < env["w"] and 0 <= c[1] < env["h"]) or c in env["blocked"] for c in path):
        return False, "blocked or out of bounds"
    missing = [k for k in env["keys"] if k not in path[:-1]]
    if missing:
        return False, f"misses key {missing[0]}"
    best = _optimum(env)
    return len(path) - 1 == best, f"{len(path) - 1} steps vs optimum {best}"


def keys_canonical(p, meta):
    from grader import _cells as cells_of

    conv = p.get("convention", "col_row")
    parts = []
    for q in p["gt_path_parts"]:
        v = resolve(meta, q)
        parts += v if q.endswith("segments_to_keys") else [v]
    out = []
    for part in parts:
        cells = cells_of(part, conv)
        out += cells if not out else cells[1:]
    return [list(c) for c in out]


def keys_wrong(p, meta):
    return keys_canonical(p, meta)[:-1]


def _kit(prefix):
    g = globals()
    return {"grade": g[f"{prefix}_grade"], "canonical": g[f"{prefix}_canonical"], "wrong": g[f"{prefix}_wrong"]}


ORDERED = {"grade": ordered_points_grade, "canonical": ordered_points_canonical, "wrong": ordered_points_wrong}
MATCHED = {"grade": matched_moves_grade, "canonical": matched_moves_canonical, "wrong": matched_moves_wrong}

CUSTOM = {
    "G-174_arrange_circles_by_circumference_data-generator": ORDERED,
    "G-24_separate_objects_no_spin_data-generator": MATCHED,
    # group 1
    "G-138_spot_unique_non_repeated_color_data-generator": _kit("polygon"),
    "G-25_seperate_object_spinning_data-generator": MATCHED,
    "G-3_stable_sort_data-generator": _kit("sorted_line"),
    "G-45_key_door_matching_data-generator": _kit("via_waypoint"),
    "G-5_multi_object_placement_data-generator": MATCHED,
    "G-9_identify_objects_in_region_data-generator": _kit("masked_set"),
    # group 2
    "O-14_shape_scale_then_outline_data-generator": _kit("o14"),
    "O-29_ballcolor_data-generator": _kit("o29"),
    "O-30_bookshelf_data-generator": _kit("o30"),
    "O-31_ball_eating_data-generator": _kit("o31"),
    "O-32_rolling_ball_data-generator": _kit("o32"),
    "O-34_dot_to_dot_task_data-generator": _kit("o34"),
    "O-45_sequence_completion_data-generator": _kit("o45"),
    # group 4
    "G-250_color_triple_intersection_red_data-generator": _kit("bbox"),
    "G-47_multiple_keys_for_one_door_data-generator": _kit("keys"),
    "G-54_connecting_color_data-generator": _kit("pairs"),
    "O-27_move_2_object_to_2_target_data-generator": MATCHED,
    "O-46_shape_sorter_data-generator": MATCHED,
    "O-56_raven_data-generator": _kit("raven"),
    "O-64_animal_matching_data-generator": MATCHED,
    "O-65_animal_size_sorting_data-generator": ORDERED,
}
