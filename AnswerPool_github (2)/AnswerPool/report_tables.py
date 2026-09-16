#!/usr/bin/env python
# ============================================================================
# report_tables.py — the paper's tables for a benchmark, from *.results.jsonl.
#
# Reproduces the layouts of Project_5_report.pdf:
#   Table 3   de-saturation: MCQ | group-scored MCQ | matching exact,
#             spread, guessing floor, closed-book (image-withheld) group rate
#   Table 4   withhold arm (p=0.4): accuracy | false answer | false abstention
#             | strict exact
#   Table 7   closed-book layout: choices-only | per-question matching, MCQ |
#             group-level matching exact, group MCQ   (+ alpha, Spearman rho)
#   Table 8   one row per benchmark: difficulty range (paired matching-minus-
#             MCQ per-question, CI), alpha, FA best / worst
#
# Expects the files run_report.sh writes under runs/<bench>/:
#   <bench>_match.<model>.results.jsonl    matching, withhold p     (Table 4)
#   <bench>_match0.<model>.results.jsonl   matching, no withhold    (Table 3/7)
#   <bench>_mcq.<model>.results.jsonl      mirror-paired MCQ
#   <bench>_co.<model>.results.jsonl       choices-only             (alpha)
#   <bench>_noimg.<model>.results.jsonl    image-withheld MCQ       (closed book)
#
#   python report_tables.py --bench scienceqa --dir runs/scienceqa
#   python report_tables.py --all --dir runs          # every bench + Table 8
# ============================================================================
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import compare as C

PRETTY = {"gpqa_diamond": "GPQA Diamond", "mmmu": "MMMU", "mmmu_pro": "MMMU-Pro",
          "mmmu_pro10": "MMMU-Pro (10-opt)", "mathvista": "MathVista",
          "scienceqa": "ScienceQA", "hellaswag": "HellaSwag"}
MODEL_PRETTY = {"claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
                "claude-fable-5-1": "Claude Fable 5.1", "gpt-5.2": "GPT-5.2",
                "gpt-5": "GPT-5", "kimi-k3": "Kimi K3",
                "gemini-2.5-flash": "Gemini 2.5 Flash", "gemini-2.5-pro": "Gemini 2.5 Pro"}


def pretty(m):
    return MODEL_PRETTY.get(m, m)


# ----------------------------------------------------------------- loading --
def load_all(d, bench):
    """-> {model: {arm: recs}}. Filename: <bench>_<arm>.<model>.results.jsonl"""
    out = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(d, f"{bench}_*.results.jsonl"))):
        base = os.path.basename(p)[len(bench) + 1:-len(".results.jsonl")]
        arm, model = base.split(".", 1)
        recs = C.load(p)
        if recs:
            out[model][arm] = recs
    return out


# ----------------------------------------------------------------- metrics --
def per_pair_nm(recs):
    """(accuracy on present slots, n, per-group hit lists)"""
    return C.per_pair(recs)


def false_answer(recs):
    """withheld slots answered with a letter: confabulation"""
    ans = tot = 0
    for r in recs:
        for i in r.get("withheld", []):
            if i < len(r["pred"]):
                p = r["pred"][i]
                if p is None:
                    continue
                tot += 1
                ans += (p != "none")
    return (ans / tot if tot else float("nan")), tot


def false_abstention(recs):
    """present slots answered 'none': over-refusal"""
    ab = tot = 0
    for r in recs:
        wh = set(r.get("withheld", []))
        for i, p in enumerate(r["pred"]):
            if i in wh or p is None:
                continue
            tot += 1
            ab += (p == "none")
    return (ab / tot if tot else float("nan")), tot


def unparsed(recs):
    tot = sum(len(r["pred"]) for r in recs)
    un = sum(1 for r in recs for p in r["pred"] if p is None)
    return un, tot


def nm(recs):
    ns = [r.get("n") for r in recs if r.get("n")]
    ms = [r.get("m") for r in recs if r.get("m")]
    n = max(set(ns), key=ns.count) if ns else 0
    return n, (min(ms) if ms else 0), (sorted(ms)[len(ms)//2] if ms else 0), (max(ms) if ms else 0)


def floor_exact(n, m):
    return 1.0 / math.prod(range(m - n + 1, m + 1)) if m >= n >= 1 else float("nan")


def alpha(co_acc, n, m):
    """separability: (observed - 1/M) / (1/N - 1/M), Section 4.5"""
    lo, cap = 1.0 / m, 1.0 / n
    return (co_acc - lo) / (cap - lo) if cap > lo else float("nan")


def ci_vs_floor(recs, floor, iters=2000):
    """is choices-only significantly above 1/M? bootstrap over groups"""
    _, _, by = C.per_pair(recs)
    keys = sorted(by)
    if not keys:
        return False
    import random
    rng = random.Random(0)
    lows = []
    for _ in range(iters):
        s = [keys[rng.randrange(len(keys))] for _ in keys]
        h = [x for k in s for x in by[k]]
        lows.append(sum(h) / len(h))
    lows.sort()
    return lows[int(0.025 * len(lows))] > floor


# ------------------------------------------------------------------ tables --
def f4(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def sci(x):
    if x is None or math.isnan(x):
        return "–"
    e = int(math.floor(math.log10(x))) if x > 0 else 0
    return f"{x/10**e:.1f}e{e}" if x < 1e-3 else f"{x:.4f}"


def md_table(header, rows, caption=""):
    out = [f"**{caption}**\n"] if caption else []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def tex_table(header, rows, caption="", label=""):
    cols = "l" + "c" * (len(header) - 1)
    out = ["\\begin{table}[t]\\centering", f"\\caption{{{caption}}}",
           f"\\label{{tab:{label}}}" if label else "", f"\\begin{{tabular}}{{{cols}}}",
           "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(str(x).replace("%", "\\%") for x in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(x for x in out if x) + "\n"


def bench_tables(bench, data):
    """-> (markdown, latex, summary row for Table 8)"""
    md, tex = [], []
    name = PRETTY.get(bench, bench)
    models = sorted(data)
    if not models:
        return "", "", None

    # ---- Table 7-style main table (closed-book layout fits every benchmark
    #      here: GPQA has no passage; visual ones treat the image as passage
    #      and report the image-withheld rate in the last column)
    rows7 = []
    diff_ranges = []
    mcq_acc, match_acc = {}, {}
    alphas = {}
    for m in models:
        arms = data[m]
        match_arm = arms.get("match0") or arms.get("match")
        r = [pretty(m)]
        # choices-only
        if "co" in arms:
            co_acc, _, _ = C.per_pair(arms["co"])
            n, mmin, mmed, mmax = nm(arms["co"])
            alphas[m] = alpha(co_acc, n, mmed) if n and mmed else float("nan")
            r.append(f4(co_acc))
        else:
            r.append("–")
        # per-question
        if match_arm:
            a, _, _ = C.per_pair(match_arm)
            match_acc[m] = a
            r.append(f4(a))
        else:
            r.append("–")
        if "mcq" in arms:
            a, _, _ = C.per_pair(arms["mcq"])
            mcq_acc[m] = a
            r.append(f4(a))
        else:
            r.append("–")
        # group level
        if match_arm:
            ex, _ = C.exact(match_arm)
            r.append(f4(ex))
        else:
            r.append("–")
        if "mcq" in arms:
            gb, _, _ = C.exact_blocked(arms["mcq"])
            r.append(f4(gb))
        else:
            r.append("–")
        # closed-book / image-withheld, group level
        if "noimg" in arms:
            gb, _, _ = C.exact_blocked(arms["noimg"])
            r.append(f4(gb))
        else:
            r.append("n/a")
        # unparsed: no grammar on API backends, must be reported (Sec. 3.5)
        un, tot = unparsed(match_arm) if match_arm else (0, 0)
        r.append(f"{un}/{tot}")
        rows7.append(r)
        # paired difficulty: matching - mcq per question, groups resampled
        if match_arm and "mcq" in arms:
            _, _, g1 = C.per_pair(match_arm)
            _, _, g2 = C.per_pair(arms["mcq"])
            pb = C.paired_bootstrap(g1, g2)
            if pb:
                diff_ranges.append((m, pb))

    n, mmin, mmed, mmax = nm(next(iter(data.values())).get("match0")
                             or next(iter(data.values())).get("match") or [])
    fl_mcq = None
    if any("mcq" in a for a in data.values()):
        mm = [r["m"] for a in data.values() if "mcq" in a for r in a["mcq"] if r.get("m")]
        fl_mcq = 1.0 / (sum(mm) / len(mm)) if mm else None
    hdr7 = ["", "Choices-only", "Matching (per-q)", "MCQ (per-q)",
            "Match. exact", "Grp. MCQ", "Image-withheld grp.", "unparsed"]
    cap7 = (f"{name}, N={n} groups, pool M={mmed}"
            + (f" (range {mmin}–{mmax})" if mmin != mmax else "")
            + f". Choices-only floor 1/M={1/mmed:.3f} with cap 1/N={1/n:.3f}. "
              f"MCQ floor {fl_mcq:.2f}, exact-assignment floor {sci(floor_exact(n, mmin))}."
            if n and mmed else f"{name}")
    md.append(md_table(hdr7, rows7, f"Table 7-style: {cap7}"))
    tex.append(tex_table(hdr7, rows7, cap7, f"{bench}_main"))

    # ---- Table 3-style: de-saturation summary
    if mcq_acc and match_acc:
        rows3 = []
        for m in models:
            if m in mcq_acc and m in match_acc:
                gb, _, _ = C.exact_blocked(data[m]["mcq"])
                ex, _ = C.exact(data[m].get("match0") or data[m]["match"])
                rows3.append([pretty(m), f4(mcq_acc[m]), f4(gb), f4(ex)])
        ms_ = [m for m in models if m in mcq_acc and m in match_acc]
        if len(ms_) >= 2:
            sp = lambda d: max(d[m] for m in ms_) - min(d[m] for m in ms_)
            gbs = {m: C.exact_blocked(data[m]["mcq"])[0] for m in ms_}
            exs = {m: C.exact(data[m].get("match0") or data[m]["match"])[0] for m in ms_}
            rows3.append(["Best-to-worst spread", f"{sp(mcq_acc):.2f}", f"{sp(gbs):.2f}",
                          f"{sp(exs):.2f}"])
            rho = C.spearman([mcq_acc[m] for m in ms_], [exs[m] for m in ms_])
            rows3.append(["Spearman ρ (vs MCQ)", "", "",
                          f"{rho:.2f}" if not math.isnan(rho) else "–"])
        rows3.append(["Guessing floor", f"{fl_mcq:.2f}" if fl_mcq else "–",
                      sci(fl_mcq ** n) if fl_mcq and n else "–",
                      sci(floor_exact(n, mmin)) if n and mmin else "–"])
        hdr3 = ["", "MCQ", f"Group-scored MCQ (×{n})", "Matching (exact)"]
        src = "match0" if any("match0" in a for a in data.values()) else "match"
        note = ("" if src == "match0" else
                " Matching exact computed on the withhold arm's present slots "
                "(no match0 arm found); build match0 for the paper's construction.")
        md.append(md_table(hdr3, rows3, f"Table 3-style: de-saturation on {name} (N={n}).{note}"))
        tex.append(tex_table(hdr3, rows3, f"De-saturation on {name} ($N$={n}).{note}",
                             f"{bench}_desat"))

    # ---- Table 4-style: withhold arm
    rows4 = []
    fa_by = {}
    for m in models:
        if "match" not in data[m]:
            continue
        wh = data[m]["match"]
        if not any(r.get("withheld") for r in wh):
            continue
        acc, _, _ = C.per_pair(wh)
        fa, nfa = false_answer(wh)
        fab, _ = false_abstention(wh)
        st, _ = C.exact_strict(wh)
        fa_by[m] = fa
        rows4.append([pretty(m), f4(acc), f4(fa), f4(fab), f4(st), str(nfa)])
    if rows4:
        p = None
        wh0 = next(a["match"] for a in data.values() if "match" in a)
        k = sum(len(r.get("withheld", [])) for r in wh0)
        tot = sum(len(r["gold"]) for r in wh0)
        p = k / tot if tot else 0
        hdr4 = ["", "Accuracy", "False answer ↓", "False abstention", "Strict ↑",
                "unanswerable n"]
        cap4 = (f"Table 4-style: withhold arm on {name} at p={p:.1f} "
                f"({int(round(p*n))} of {n} golds deleted)")
        md.append(md_table(hdr4, rows4, cap4))
        tex.append(tex_table(hdr4, rows4, cap4, f"{bench}_withhold"))

    # ---- difficulty contrasts (paired bootstrap), alpha significance
    if diff_ranges:
        lines = ["**Paired matching − MCQ per-question (groups resampled, 95% CI):**\n"]
        for m, (d, lo, hi, k) in diff_ranges:
            sig = "" if lo <= 0 <= hi else " *"
            lines.append(f"- {pretty(m)}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] over {k} groups{sig}")
        md.append("\n".join(lines) + "\n")
    if alphas:
        lines = ["**Separability α on choices-only (Proposition 1):**\n"]
        for m in models:
            if m in alphas and "co" in data[m]:
                n_, _, mmed_, _ = nm(data[m]["co"])
                sig = ci_vs_floor(data[m]["co"], 1.0 / mmed_) if mmed_ else False
                lines.append(f"- {pretty(m)}: α = {alphas[m]:.2f} "
                             f"({'sig. above 1/M' if sig else 'n.s.'})")
        md.append("\n".join(lines) + "\n")

    # ---- summary row for Table 8
    row8 = None
    if diff_ranges:
        ds = [d for _, (d, lo, hi, k) in diff_ranges]
        all_sig = all(not (lo <= 0 <= hi) for _, (d, lo, hi, k) in diff_ranges)
        arng = [v for v in alphas.values() if not math.isnan(v)]
        fas = [v for v in fa_by.values() if not math.isnan(v)]
        row8 = [name,
                f"{min(ds):+.2f} to {max(ds):+.2f}" + ("" if all_sig else " (some CI incl. 0)"),
                (f"{min(arng):.2f}–{max(arng):.2f}" if len(arng) > 1 else
                 f"{arng[0]:.2f}" if arng else "–"),
                (f"{min(fas):.2f} / {max(fas):.2f}" if fas else "–"),
                str(len(models))]
    return "\n".join(md), "\n".join(tex), row8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="")
    ap.add_argument("--dir", default="runs")
    ap.add_argument("--all", action="store_true", help="every <dir>/<bench>/ subfolder")
    ap.add_argument("--out", default="", help="write <out>.md and <out>.tex")
    a = ap.parse_args()

    benches = []
    if a.all:
        for sub in sorted(os.listdir(a.dir)):
            if os.path.isdir(os.path.join(a.dir, sub)):
                benches.append((sub, os.path.join(a.dir, sub)))
    elif a.bench:
        benches.append((a.bench, a.dir))
    else:
        raise SystemExit("--bench <name> or --all")

    md_all, tex_all, rows8 = [], [], []
    for bench, d in benches:
        data = load_all(d, bench)
        if not data:
            print(f"[{bench}] no results in {d}")
            continue
        md, tex, row8 = bench_tables(bench, data)
        md_all.append(f"## {PRETTY.get(bench, bench)}\n\n{md}")
        tex_all.append(tex)
        if row8:
            rows8.append(row8)
    if len(rows8) > 1:
        hdr8 = ["", "Difficulty (match − MCQ)", "α", "FA best / worst", "models"]
        md_all.append(md_table(hdr8, rows8, "Table 8-style: the transform across benchmarks"))
        tex_all.append(tex_table(hdr8, rows8, "The transform across benchmarks.", "generality"))
    text = "\n".join(md_all)
    print(text)
    if a.out:
        open(a.out + ".md", "w", encoding="utf-8").write(text)
        open(a.out + ".tex", "w", encoding="utf-8").write("\n".join(tex_all))
        print(f"-> {a.out}.md  {a.out}.tex")


if __name__ == "__main__":
    main()
