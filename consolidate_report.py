#!/usr/bin/env python
# ============================================================================
# consolidate_report.py — read every MMMU-Pro results.jsonl this session
# produced and print ONE table, computed directly by compare.py's own
# functions (per_pair, exact, exact_blocked, paired_bootstrap, ...). No
# number here is retyped from a chat log; every value is recomputed fresh
# from the raw per-item result files on disk.
#
#   python consolidate_report.py --bench mmmu_pro --dir .
#   python consolidate_report.py --bench mmmu_pro --dir . --model Qwen_Qwen2.5-VL-7B-Instruct
# ============================================================================
import argparse
import glob
import math
import os

import compare as C


def false_answer(recs):
    ans = tot = 0
    for r in recs:
        for i in r.get("withheld", []):
            if i < len(r["pred"]):
                p = r["pred"][i]
                if p is None:
                    continue
                tot += 1
                ans += (p != "none")
    return (ans / tot if tot else float("nan")), ans, tot


def false_abstention(recs):
    ab = tot = 0
    for r in recs:
        wh = set(r.get("withheld", []))
        for i, p in enumerate(r["pred"]):
            if i in wh or p is None:
                continue
            tot += 1
            ab += (p == "none")
    return (ab / tot if tot else float("nan")), ab, tot


def unparsed(recs):
    un = sum(1 for r in recs for p in r["pred"] if p is None)
    tot = sum(len(r["pred"]) for r in recs)
    return un, tot


def floor_exact(n, m):
    return 1.0 / math.prod(range(m - n + 1, m + 1)) if m and m >= n >= 1 else float("nan")


def f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--model", default="", help="filter to one model substring; "
                    "default picks whichever model has the most result files")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, f"{a.bench}_*.results.jsonl")))
    if not files:
        raise SystemExit(f"no {a.bench}_*.results.jsonl found under {a.dir}")

    # group by (arm, model)
    by = {}
    for p in files:
        base = os.path.basename(p)[len(a.bench) + 1:-len(".results.jsonl")]
        arm, model = base.split(".", 1)
        if a.model and a.model not in model:
            continue
        by.setdefault(model, {})[arm] = p
    if not by:
        raise SystemExit(f"no files matched --model {a.model!r}")
    model = a.model or max(by, key=lambda m: len(by[m]))
    arms = by[model]
    print(f"model: {model}")
    print(f"arms found: {sorted(arms)}\n")

    rows = []

    def add(cat, metric, val, nd=4, extra=""):
        rows.append((cat, metric, f(val, nd) if isinstance(val, float) else str(val), extra))

    match = C.load(arms["match"]) if "match" in arms else None
    mcq = C.load(arms["mcq"]) if "mcq" in arms else None
    co = C.load(arms["co"]) if "co" in arms else None
    wh = C.load(arms["wh"]) if "wh" in arms else (match if match and any(r.get("withheld") for r in match) else None)
    noimg = C.load(arms["noimg"]) if "noimg" in arms else None

    n = m = None
    if match:
        n = max(set(r["n"] for r in match if r.get("n")), key=lambda x: sum(1 for r in match if r.get("n") == x))
        ms = [r["m"] for r in match if r.get("m")]
        m = min(ms) if ms else None  # worst-case for the floor

    if mcq is not None:
        acc, tot, _ = C.per_pair(mcq)
        add("De-saturation", "MCQ accuracy", acc)
        gb, ng, _ = C.exact_blocked(mcq)
        add("De-saturation", "Group-scored MCQ (exact, blocked)", gb)
        if n:
            add("Guessing floors", "Group-scored MCQ floor", (1.0 / 4) ** n, nd=6)

    if match is not None and not any(r.get("withheld") for r in match):
        acc, _, _ = C.per_pair(match)
        add("De-saturation", "Matching (per-question)", acc)
        ex, ng = C.exact(match)
        add("De-saturation", "Matching (exact assignment)", ex)
        if n and m:
            add("Guessing floors", "Matching exact floor", floor_exact(n, m), nd=10)
        if mcq is not None:
            _, _, g1 = C.per_pair(match)
            _, _, g2 = C.per_pair(mcq)
            pb = C.paired_bootstrap(g1, g2)
            if pb:
                d, lo, hi, k = pb
                add("Statistical test", "Matching-MCQ contrast", d)
                add("Statistical test", "95% CI", f"[{lo:.4f}, {hi:.4f}]", nd=0)
                add("Statistical test", "n groups / significant", f"{k} / {'yes' if not (lo<=0<=hi) else 'no'}", nd=0)

    if co is not None:
        acc, tot, _ = C.per_pair(co)
        add("Choices-only control", "Accuracy", acc)
        if n and m:
            alpha = (acc - 1.0 / m) / (1.0 / n - 1.0 / m) if (1.0 / n - 1.0 / m) else float("nan")
            add("Choices-only control", "Separability alpha", alpha)

    if wh is not None and any(r.get("withheld") for r in wh):
        acc, _, _ = C.per_pair(wh)
        add("Confabulation (withholding)", "Per-pair accuracy (present slots)", acc)
        fa, fan, fat = false_answer(wh)
        add("Confabulation (withholding)", "False-answer rate", fa)
        add("Confabulation (withholding)", "False-answer count", f"{fan}/{fat}", nd=0)
        fb, fbn, fbt = false_abstention(wh)
        add("Confabulation (withholding)", "Miss / false-abstention rate", fb)
        add("Confabulation (withholding)", "Miss count", f"{fbn}/{fbt}", nd=0)
        ex, ng = C.exact(wh)
        add("Confabulation (withholding)", "Exact assignment (withhold arm)", ex)

    if mcq is not None and noimg is not None:
        acc_img, _, _ = C.per_pair(mcq)
        acc_noimg, _, _ = C.per_pair(noimg)
        add("Image dependency", "MCQ accuracy WITH image", acc_img)
        add("Image dependency", "MCQ accuracy WITHOUT image", acc_noimg)
        add("Image dependency", "Drop from removing image", acc_img - acc_noimg)
        add("Image dependency", "Above-chance residual (no image)", acc_noimg - 0.25)

    if match is not None:
        un, tot = unparsed(match)
        add("Coverage", "Unparsed slots (match arm)", f"{un}/{tot}", nd=0)
    if n and m:
        add("Coverage", "N, M (pool size)", f"N={n}, M={m}", nd=0)

    # print
    cats_seen = []
    for cat, metric, val, _ in rows:
        if cat not in cats_seen:
            cats_seen.append(cat)
    w1 = max(len(c) for c in cats_seen) + 2
    w2 = max(len(m) for _, m, _, _ in rows) + 2
    print(f"{'Category':<{w1}}{'Metric':<{w2}}Value")
    print("-" * (w1 + w2 + 20))
    last_cat = None
    for cat, metric, val, _ in rows:
        show_cat = cat if cat != last_cat else ""
        print(f"{show_cat:<{w1}}{metric:<{w2}}{val}")
        last_cat = cat


if __name__ == "__main__":
    main()
