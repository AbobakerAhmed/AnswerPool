#!/usr/bin/env python
# ============================================================================
# collect_quality.py -- summarise the QuALITY batch: headline cells, the N sweep
# rebuilt on the headline groups, and the seed / reuse / wording ablations.
#
#   python -m AnswerPooling.collect_quality            # reads *.results.jsonl in the cwd
#   python -m AnswerPooling.collect_quality --dir runs # elsewhere
#
# Every number is a point estimate over all groups in the file. Paired
# differences between two settings are computed on shared group ids and
# come with a 95% bootstrap interval over groups (2,000 resamples).
# ============================================================================
import argparse
import glob
import json
import os
import random
from collections import defaultdict

STEMS = ["h_match", "h_mcq", "h_co", "h_match_n3", "h_match_n4", "h_match_n5", "h_match_n6",
         "h_match_seed1", "h_match_reuse", "h_wh40", "wh40_alt", "wh40_reuse",
         "h_prose40", "h_mcqnone40", "h_easy"]

SHORT = {  # pretty names, longest key first so FP8 wins over the bare 32B
    "Qwen_Qwen3-32B-FP8": "Qwen3-32B", "Qwen_Qwen3-32B": "Qwen3-32B",
    "Qwen_Qwen3-8B": "Qwen3-8B", "Qwen_Qwen2.5-7B": "Qwen2.5-7B",
    "google_gemma-3-27b-it": "Gemma 3 27B", "google_gemma-3-12b-it": "Gemma 3 12B",
    "gemini-3.6-flash": "Gemini 3.6 Flash", "gemini-3.1-flash-lite": "Gemini 3.1 F-Lite",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemma-4-26b-a4b-it-maas": "Gemma 4 26B",
    "google_gemma-4-26b-a4b-it": "Gemma 4 26B",   # local vLLM run, same weights
    "openai_gpt-oss-120b": "gpt-oss-120b", "openai_gpt-oss-20b": "gpt-oss-20b",
    "Qwen_Qwen3.5-27B": "Qwen3.5-27B", "Qwen_Qwen3.5-4B": "Qwen3.5-4B",
}


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def model_of(path, stem):
    base = os.path.basename(path)
    mid = base[len(stem) + 1: -len(".results.jsonl")]
    return SHORT.get(mid, mid)


def metrics(recs, keep=None):
    """Per-file point estimates. Groups are grouped by group_id so that
    per-item MCQ files score all-correct the same way pooled files do.
    keep: optional set of group ids, everything else is dropped, so every
    model is scored on identical groups."""
    by_g = defaultdict(list)
    for r in recs:
        if keep is not None and r["group_id"] not in keep:
            continue
        by_g[r["group_id"]].append(r)
    slots_ok = slots = 0        # per-question, present-answer slots only
    fa = fa_tot = 0             # removed-answer slots answered anyway
    fab = fab_tot = 0           # present-answer slots declined
    allc = 0                    # every slot right, none included
    unp = err = 0
    per_group_allc = {}
    per_group_fa = {}           # fraction of a group's removed slots answered anyway
    for gid, rs in by_g.items():
        ok_all = True
        g_fa = g_fa_tot = 0
        for r in rs:
            if r.get("err"):
                err += 1
            preds, golds = r["pred"], r["gold"]
            if len(preds) != len(golds) or any(p is None for p in preds):
                unp += 1
                ok_all = False
            for p, g in zip(preds, golds):
                if g == "none":
                    fa_tot += 1
                    fa += (p != "none")
                    g_fa_tot += 1
                    g_fa += (p != "none")
                else:
                    slots += 1
                    slots_ok += (p == g)
                    fab_tot += 1
                    fab += (p == "none")
                ok_all &= (p == g)
        allc += ok_all
        per_group_allc[gid] = ok_all
        if g_fa_tot:
            per_group_fa[gid] = g_fa / g_fa_tot
    n = len(by_g)
    return dict(
        groups=n, unparsed=unp, errors=err,
        acc=slots_ok / slots if slots else float("nan"),
        allc=allc / n if n else float("nan"),
        fa=fa / fa_tot if fa_tot else None,
        fab=fab / fab_tot if fab_tot else None,
        _g=per_group_allc,
        _fa=per_group_fa,
    )


def paired_boot(a, b, key="_g", reps=2000, seed=0):
    """Paired difference b - a on shared group ids, all-correct indicator."""
    ga, gb = a[key], b[key]
    shared = sorted(set(ga) & set(gb))
    if not shared:
        return None
    d = [float(gb[g]) - float(ga[g]) for g in shared]
    rng = random.Random(seed)
    mean = sum(d) / len(d)
    boots = []
    for _ in range(reps):
        s = [d[rng.randrange(len(d))] for _ in d]
        boots.append(sum(s) / len(s))
    boots.sort()
    return mean, boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1], len(shared)


def fmt(x, nd=3):
    return "  --  " if x is None or x != x else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--pair-to", default="",
                    help="a results file; the headline block is restricted to "
                         "the group ids it contains, e.g. an API model's h_match "
                         "results that ran with --limit 300")
    ap.add_argument("--legacy-wh40", default="",
                    help="stem of an earlier removed-answers file, printed "
                         "for reference beside the paired ablations")
    a = ap.parse_args()

    keep = None
    if a.pair_to:
        keep = {r["group_id"] for r in load(a.pair_to)}
        print(f"headline block restricted to {len(keep)} groups from {a.pair_to}")

    R = defaultdict(dict)      # R[stem][model] = metrics
    P = defaultdict(dict)      # same, restricted to the paired group set
    RAW = defaultdict(dict)    # RAW[stem][model] = records, for the abstention pairing
    for stem in STEMS:
        for f in sorted(glob.glob(os.path.join(a.dir, f"{stem}.*.results.jsonl"))):
            if ".judged." in f:
                continue
            recs = load(f)
            R[stem][model_of(f, stem)] = metrics(recs)
            if stem in ("h_wh40", "h_prose40", "h_mcqnone40"):
                RAW[stem][model_of(f, stem)] = recs
            if keep is not None and stem in ("h_match", "h_mcq"):
                P[stem][model_of(f, stem)] = metrics(recs, keep)
    T2 = P if keep is not None else R

    # The abstention block must put every model on identical questions and
    # removed answers: some models ran h_wh40 with --limit 300, so restrict all
    # three formats to the group ids every h_wh40 file shares (the mirror-built
    # formats share ids).
    W = defaultdict(dict)
    if RAW["h_wh40"]:
        keep_w = set.intersection(*[{r["group_id"] for r in recs}
                                    for recs in RAW["h_wh40"].values()])
        print(f"abstention block restricted to {len(keep_w)} groups shared by every h_wh40 file")
        for stem in ("h_wh40", "h_prose40", "h_mcqnone40"):
            for m, recs in RAW[stem].items():
                W[stem][m] = metrics(recs, keep_w)

    models = sorted({m for s in R.values() for m in s})
    if not models:
        raise SystemExit("no result files found; check --dir and file names")

    print("=" * 78)
    print("HEALTH  (unparsed and errors must be 0)")
    print("=" * 78)
    for stem in STEMS:
        for m in models:
            x = R[stem].get(m)
            if x and (x["unparsed"] or x["errors"]):
                print(f"  {stem:16s} {m:20s} unparsed={x['unparsed']} errors={x['errors']}")
    print("  (nothing above this line means all clean)")
    print()
    print("  groups per file (a mismatch here means different builds):")
    for stem in STEMS:
        counts = sorted({x["groups"] for x in R[stem].values()})
        if counts:
            print(f"    {stem:16s} {counts}")

    print("\n" + "=" * 78)
    print("HEADLINE, QuALITY   MCQ / pooled per question | MCQ all-correct / pooled all-correct")
    print("=" * 78)
    for m in models:
        mc, po = T2["h_mcq"].get(m), T2["h_match"].get(m)
        if not (mc and po):
            continue
        print(f"  {m:20s} {fmt(mc['acc'])} / {fmt(po['acc'])}   |   "
              f"{fmt(mc['allc'])} / {fmt(po['allc'])}    (groups {po['groups']})")

    print("\n" + "=" * 78)
    print("N SWEEP on the headline groups   all-correct (per-question)")
    print("=" * 78)
    print(f"  {'model':20s} {'N=3':>16s} {'N=4':>16s} {'N=5':>16s} {'N=6':>16s}")
    for m in models:
        cells = []
        for stem in ("h_match_n3", "h_match_n4",
                     "h_match_n5" if R["h_match_n5"] else "h_match", "h_match_n6"):
            x = R[stem].get(m)
            cells.append(f"{fmt(x['allc'])} ({fmt(x['acc'])})" if x else "       --       ")
        if any(c.strip() != "--" for c in cells):
            print(f"  {m:20s} " + " ".join(f"{c:>16s}" for c in cells))

    def ablation(name, base_stem, alt_stem, what, paired=True):
        print("\n" + "=" * 78)
        print(f"{name}   {what}")
        print("=" * 78)
        for m in models:
            b, x = R[base_stem].get(m), R[alt_stem].get(m)
            if not (b and x):
                continue
            if what.startswith("all-correct"):
                pb = paired_boot(b, x) if paired else None
                ci = f"paired diff {pb[0]:+.3f}  95% [{pb[1]:+.3f}, {pb[2]:+.3f}]  k={pb[3]}" if pb else ""
                print(f"  {m:20s} {fmt(b['allc'])} -> {fmt(x['allc'])}   per-q {fmt(b['acc'])} -> {fmt(x['acc'])}   {ci}")
            else:
                pb = paired_boot(b, x, key="_fa") if paired else None
                ci = f"paired FA diff {pb[0]:+.3f}  95% [{pb[1]:+.3f}, {pb[2]:+.3f}]  k={pb[3]}" if pb else ""
                print(f"  {m:20s} false answer {fmt(b['fa'])} -> {fmt(x['fa'])}   "
                      f"false abstention {fmt(b['fab'])} -> {fmt(x['fab'])}   "
                      f"acc {fmt(b['acc'])} -> {fmt(x['acc'])}   {ci}")

    # h_match (470 groups) predates the current filter file; h_match_n5 is the
    # same-filter, same-seed build the ablations were made from, so it is the
    # only honest baseline for them.
    base = "h_match_n5" if R["h_match_n5"] else "h_match"
    ablation(f"SEED   {base} (seed 0) -> h_match_seed1", base, "h_match_seed1",
             "all-correct, same rows re-chunked, no pairing across seeds", paired=False)
    ablation(f"REUSE  exclusive -> options may be reused  (baseline {base})", base, "h_match_reuse",
             "all-correct, same groups, paired")
    ablation("WORDING  default decline line -> alt", "h_wh40", "wh40_alt",
             "false-answer / false-abstention, same groups and removed set, paired")
    ablation("REUSE UNDER REMOVAL  exclusive -> options may be reused", "h_wh40", "wh40_reuse",
             "false-answer / false-abstention, same groups and removed set, paired")

    print()
    print("=" * 78)
    print("ABSTENTION, all models on the shared groups   false answer / false abstention under the three formats")
    print("=" * 78)
    print(f"  {'model':20s} {'pooled':>15s} {'MCQ+instruction':>15s} {'MCQ+none':>15s}   acc(pooled)")
    T3 = W if W else R
    for m in models:
        cells = []
        for stem in ("h_wh40", "h_prose40", "h_mcqnone40"):
            x = T3[stem].get(m)
            cells.append(f"{fmt(x['fa'])} / {fmt(x['fab'])}" if x else "      --       ")
        w = T3["h_wh40"].get(m)
        if any(c.strip() != "--" for c in cells):
            print(f"  {m:20s} " + " ".join(f"{c:>15s}" for c in cells)
                  + (f"   {fmt(w['acc'])}" if w else ""))

    if R["h_co"]:
        print()
        print("=" * 78)
        print("OPTIONS ONLY   accuracy on the pool alone, floor 0.05, cap 0.20, alpha = (acc - 0.05) / 0.15")
        print("=" * 78)
        for m in models:
            x = R["h_co"].get(m)
            if x:
                alpha = max(0.0, (x["acc"] - 0.05) / 0.15)
                print(f"  {m:20s} {fmt(x['acc'])}   alpha {alpha:.2f}   groups {x['groups']}")

    print()
    print("=" * 78)
    print("UNRELATED POOL   MCQ per-question -> pooled on unrelated-gold pools")
    print("=" * 78)
    for m in models:
        mc, e = R["h_mcq"].get(m), R["h_easy"].get(m)
        if e:
            print(f"  {m:20s} MCQ {fmt(mc['acc']) if mc else '  --  '} -> unrelated pool "
                  f"{fmt(e['acc'])}   all-correct {fmt(e['allc'])}   groups {e['groups']}")

    if a.legacy_wh40:
        print()
        print("=" * 78)
        print(f"REFERENCE  {a.legacy_wh40} (an earlier removed-answers file, its own removed set)")
        print("=" * 78)
        for f in sorted(glob.glob(os.path.join(a.dir, f"{a.legacy_wh40}.*.results.jsonl"))):
            x = metrics(load(f))
            print(f"  {model_of(f, a.legacy_wh40):20s} false answer {fmt(x['fa'])}   "
                  f"false abstention {fmt(x['fab'])}   acc {fmt(x['acc'])}   groups {x['groups']}")

    print("\nread: the N=5 sweep cell must now equal the headline pooled all-correct; "
          "reuse tells you whether exclusivity carries the difficulty (all-correct "
          "rises when the rule is dropped) or not; wording should move the "
          "false-answer rate by little.")


if __name__ == "__main__":
    main()
