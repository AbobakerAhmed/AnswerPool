#!/usr/bin/env python
# ============================================================================
# compare.py -- per-model tables from *.results.jsonl, optionally without the
# groups the ambiguity check flags (--exclude-groups).
#
# Reports:
#   * per-model accuracy per arm
#   * model rankings under MCQ and pooling, and Spearman rho between them
#   * passage contribution (full - blind)/(1 - blind) per model
#   * paired bootstrap CI on the MCQ-vs-pooled gap (item-level pairing)
#
#   python -m AnswerPooling.compare --glob "*.results.jsonl"
#   python -m AnswerPooling.compare --glob "h_match.*.results.jsonl" --exclude-groups flagged_groups.txt
# ============================================================================
import argparse
import glob
import json
import os
import random
from collections import defaultdict


EXCLUDE = set()          # group_ids to drop everywhere, set by --exclude-groups


def load(path):
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                if r.get("group_id") not in EXCLUDE:
                    recs.append(r)
            except Exception:
                pass
    return recs


def per_pair(recs):
    """-> (accuracy, n_scored, {group_id: [bools]}) ignoring withheld slots."""
    ok = tot = 0
    by_group = defaultdict(list)
    for r in recs:
        wh = set(r.get("withheld", []))
        for i, (g, p) in enumerate(zip(r["gold"], r["pred"])):
            if i in wh or p is None:
                continue
            hit = (p == g)
            ok += hit; tot += 1
            by_group[r["group_id"]].append(hit)
    return (ok / tot if tot else 0.0), tot, by_group


def exact(recs):
    """Fraction of groups where EVERY scored slot is right. Unparsed counts as
    wrong here (unlike per-pair, which skips it): a response that omits a slot
    has not produced a valid assignment. Chance is 1/(M!/(M-N)!) -- ~5e-07 at
    N=5,M=20 -- so unlike per-pair MCQ accuracy this metric has no ceiling
    problem and stays informative once models saturate the easy format."""
    ok = tot = 0
    for r in recs:
        wh = set(r.get("withheld", []))
        slots = [(g, p) for i, (g, p) in enumerate(zip(r["gold"], r["pred"]))
                 if i not in wh]
        if not slots:
            continue
        tot += 1
        ok += all(p is not None and p == g for g, p in slots)
    return (ok / tot if tot else 0.0), tot


def exact_strict(recs):
    """Every slot right INCLUDING answering 'none' on withheld questions.

    exact() skips withheld slots, so a model that never abstains is not
    penalised there -- Qwen scores 0.1033 under exact() and 0.0000 here. On
    withhold arms this is the metric that means something: it requires getting
    the assignment right AND knowing which questions have no answer, which is
    exactly what per-question MCQ cannot ask. Off withhold arms the two are
    identical.
    """
    ok = tot = 0
    for r in recs:
        wh = set(r.get("withheld", []))
        pairs = list(enumerate(zip(r["gold"], r["pred"])))
        if not pairs:
            continue
        tot += 1
        ok += all((p == "none") if i in wh else (p is not None and p == g)
                  for i, (g, p) in pairs)
    return (ok / tot if tot else 0.0), tot


def exact_strict_blocked(recs):
    """exact_strict() for per-question arms: group by group_id, require every
    question right AND every withheld one abstained on."""
    by = defaultdict(list)
    for r in recs:
        wh = set(r.get("withheld", []))
        for i, (g, p) in enumerate(zip(r["gold"], r["pred"])):
            by[r["group_id"]].append(
                (p == "none") if i in wh else (p is not None and p == g))
    if not by:
        return 0.0, 0
    return sum(1 for v in by.values() if all(v)) / len(by), len(by)


def exact_blocked(recs):
    """MCQ scored on the SAME unit as a matching group: collect the single-question
    records sharing a group_id and require all of them correct.

    Without this, 'blind exact assignment is 1% vs MCQ blind 35%' compares a
    5-question joint event against a 1-question event and the format looks far
    more artifact-resistant than it is -- five independent MCQ answers at 0.35
    jointly land near 0.005 on their own. This measures the joint event directly
    instead of assuming independence.
    """
    by = defaultdict(list)
    for r in recs:
        wh = set(r.get("withheld", []))
        for i, (g, p) in enumerate(zip(r["gold"], r["pred"])):
            if i not in wh:
                by[r["group_id"]].append(p is not None and p == g)
    sizes = [len(v) for v in by.values()]
    if not by:
        return 0.0, 0, 0.0
    ok = sum(1 for v in by.values() if all(v))
    return ok / len(by), len(by), sum(sizes) / len(sizes)


def spearman(a, b):
    """rank correlation without scipy"""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def paired_bootstrap(g1, g2, iters=2000, seed=0):
    """CI on mean(arm1) - mean(arm2) resampling GROUPS (keeps items coupled)."""
    keys = sorted(set(g1) & set(g2))
    if not keys:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        a = [h for k in sample for h in g1[k]]
        b = [h for k in sample for h in g2[k]]
        if a and b:
            diffs.append(sum(a) / len(a) - sum(b) / len(b))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    obs_a = [h for k in keys for h in g1[k]]
    obs_b = [h for k in keys for h in g2[k]]
    return (sum(obs_a)/len(obs_a) - sum(obs_b)/len(obs_b), lo, hi, len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.results.jsonl")
    ap.add_argument("--exclude-groups", default="",
                    help="file of group_ids to drop from every table (one per "
                         "line, from verify_wellformed.py). The STABILITY check: "
                         "if the main tables barely move on the screened subset, "
                         "the ceiling is capability, not ambiguity.")
    args = ap.parse_args()
    if args.exclude_groups and os.path.exists(args.exclude_groups):
        EXCLUDE.update(l.strip() for l in open(args.exclude_groups, encoding="utf-8")
                       if l.strip())
        print(f"excluding {len(EXCLUDE)} flagged groups from every table\n")

    # filename convention: <file>.<model>.results.jsonl
    table = defaultdict(dict)          # model -> arm -> (acc, n)
    groups = defaultdict(dict)         # model -> arm -> {gid: [bools]}
    ex = defaultdict(dict)             # model -> arm -> (exact rate, n groups)
    blocked = defaultdict(dict)        # model -> mcq arm -> (all-correct, k, avg size)
    strict = defaultdict(dict)         # model -> wh arm -> (exact incl. abstention, k)
    for path in sorted(glob.glob(args.glob)):
        recs = load(path)
        if not recs:
            continue
        # ".judged.results.jsonl" -> stripping only ".results.jsonl" leaves
        # ".judged" as the last dot-field, so every judged model collapsed into
        # a single row literally named "judged".
        # freeform preds are ANSWER/ABSTAIN classifications, not answers --
        # scoring them as accuracy renders nonsense columns. Their only home
        # is the false-match block, which loads them separately.
        if recs[0].get("arm") == "freeform":
            continue
        base = os.path.basename(path).replace(".judged.results.jsonl", "") \
                                     .replace(".results.jsonl", "")
        parts = base.split(".")
        model = parts[-1] if len(parts) > 1 else "unknown"
        arm = recs[0].get("arm", "?")
        stem = parts[0]
        # distractor pooling is read from the DATA (pool larger than the number of
        # questions), not from the filename -- a stem-prefix guess mislabels any
        # file not named g_d* and silently merges it with the golds-only arm.
        # Older results files predate n/m being recorded, so fall back to the
        # source .jsonl: otherwise two models that ran the SAME arm land in
        # different columns and the ranking comparison silently finds <3 models.
        n_q, m_c = recs[0].get("n"), recs[0].get("m")
        if not m_c:
            src = os.path.join(os.path.dirname(path) or ".", stem + ".jsonl")
            if os.path.exists(src):
                for line in open(src, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        try:
                            s0 = json.loads(line)
                            n_q, m_c = s0.get("n"), s0.get("m")
                        except Exception:
                            pass
                        break
        # easy pads with cross-article golds and freeform has no real pool --
        # a large M must not stamp them "+d" (distractor-pooled)
        has_d = bool(not arm.startswith("mcq") and arm not in ("easy", "freeform")
                     and n_q and m_c and m_c > n_q)
        # The withhold FRACTION goes in the key. '+wh' alone made h_wh20 and
        # h_wh40 the same column, so whichever loaded second silently replaced
        # the other -- two different experiments, one visible number.
        fr = [len(r.get("withheld", [])) / max(1, len(r["gold"])) for r in recs]
        wh = sum(fr) / len(fr) if fr else 0.0
        key = f"{arm}{'+d' if has_d else ''}{f'+wh{round(wh*100)}' if wh else ''}"
        if key in table[model]:          # e.g. the --no-guided control
            key = f"{key}:{stem}"
        acc, n, by_g = per_pair(recs)
        table[model][key] = (acc, n)
        groups[model][key] = by_g
        ex[model][key] = exact(recs)
        if any(r.get("withheld") for r in recs):
            # mcq_none records are one question each, so exact_strict() would
            # report a per-ITEM rate against matching's per-GROUP one. Block it
            # by group first: the joint "assignment right AND abstentions right"
            # event exists in both formats and must be compared on the same unit.
            strict[model][key] = (exact_strict_blocked(recs)
                                  if arm.startswith("mcq") else exact_strict(recs))
        if arm.startswith("mcq"):
            blocked[model][key] = exact_blocked(recs)

    arms = sorted({a for m in table.values() for a in m})
    print(f"{'model':34s} " + " ".join(f"{a:>18s}" for a in arms))
    print("-" * (34 + 19 * len(arms)))
    for model in sorted(table):
        cells = []
        for a in arms:
            if a in table[model]:
                acc, n = table[model][a]
                cells.append(f"{acc:.4f} (n={n})".rjust(18))
            else:
                cells.append("-".rjust(18))
        print(f"{model:34s} " + " ".join(cells))

    # ---- exact assignment --------------------------------------------------
    print(f"\nEXACT ASSIGNMENT (every slot in the group correct; unparsed = wrong)")
    print(f"{'model':34s} " + " ".join(f"{a:>18s}" for a in arms))
    for model in sorted(ex):
        cells = []
        for a in arms:
            if a in ex[model]:
                e, k = ex[model][a]
                cells.append(f"{e:.4f} (k={k})".rjust(18))
            else:
                cells.append("-".rjust(18))
        print(f"{model:34s} " + " ".join(cells))

    # ---- exact INCLUDING abstention (withhold arms only) -------------------
    if strict:
        print("\nEXACT + ABSTENTION (every slot right AND 'none' on withheld "
              "questions)\nthe headline metric on withhold arms — accuracy alone "
              "cannot express it")
        s_arms = sorted({a for m in strict.values() for a in m})
        print(f"{'model':34s} " + " ".join(f"{a:>18s}" for a in s_arms))
        for model in sorted(strict):
            cells = []
            for a in s_arms:
                if a in strict[model]:
                    e, k = strict[model][a]
                    cells.append(f"{e:.4f} (k={k})".rjust(18))
                else:
                    cells.append("-".rjust(18))
            print(f"{model:34s} " + " ".join(cells))

    # ---- MCQ on the matching unit -----------------------------------------
    if blocked:
        print("\nMCQ SCORED PER GROUP (all questions of a group correct) — the "
              "like-for-like\ncomparison against matching's exact-assignment column above")
        b_arms = sorted({a for m in blocked.values() for a in m})
        print(f"{'model':34s} " + " ".join(f"{a:>18s}" for a in b_arms))
        for model in sorted(blocked):
            cells = []
            for a in b_arms:
                if a in blocked[model]:
                    e, k, sz = blocked[model][a]
                    cells.append(f"{e:.4f} (k={k},{sz:.1f}q)".rjust(18))
                else:
                    cells.append("-".rjust(18))
            print(f"{model:34s} " + " ".join(cells))

    # ---- false-match / miss on withheld arms ------------------------------
    fm_rows = []
    fm_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(glob.glob(args.glob)):
        recs = load(path)
        if not recs or not any(r.get("withheld") for r in recs):
            continue
        model = os.path.basename(path).replace(".results.jsonl", "").split(".")[-1]
        stem = os.path.basename(path).split(".")[0]
        fm = fmt = miss = pres = 0
        for r in recs:
            for i in set(r.get("withheld", [])):
                if i < len(r["pred"]) and r["pred"][i] is not None:
                    fm_groups[stem][model][r["group_id"]].append(
                        r["pred"][i] != "none")
        for r in recs:
            wh = set(r.get("withheld", []))
            for i, (g, p) in enumerate(zip(r["gold"], r["pred"])):
                if p is None:
                    continue
                if i in wh:
                    fmt += 1; fm += (p != "none")
                else:
                    pres += 1; miss += (p == "none")
        if fmt:
            fm_rows.append((model, os.path.basename(path).split(".")[0],
                            fm / fmt, miss / max(1, pres), fmt))
    if fm_rows:
        print("\nFALSE-MATCH (assigned an answer to a question with NO answer in the pool)")
        print(f"  {'model':32s} {'arm':12s} {'false-match':>12s} {'miss':>8s} {'n':>6s}")
        for m, a, f, ms, n in sorted(fm_rows):
            print(f"  {m:32s} {a:12s} {f:12.4f} {ms:8.4f} {n:6d}")

    # ---- is the false-match ordering real? --------------------------------
    # The models answer the SAME groups with the SAME questions withheld, so an
    # unpaired two-proportion test throws away the pairing and understates the
    # evidence. Resample groups and keep every model's outcome on that group
    # together, exactly as for the arm-vs-arm comparison above.
    for stem, per_model in sorted(fm_groups.items()):
        models = sorted(per_model)
        if len(models) < 2:
            continue
        print(f"\npaired bootstrap on FALSE-MATCH, {stem} (resampling groups)")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a_m, b_m = models[i], models[j]
                r = paired_bootstrap(per_model[a_m], per_model[b_m])
                if not r:
                    continue
                d, lo, hi, k = r
                sig = "" if lo <= 0 <= hi else "   SIGNIFICANT"
                print(f"  {a_m:26s} - {b_m:26s} {d:+.4f}  "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}]  (k={k}){sig}")

    # ---- passage contribution -------------------------------------------
    print("\npassage contribution = (full - blind) / (1 - blind)")
    for model in sorted(table):
        for full_a, blind_a in (("matching+d", "no_passage+d"),
                                ("matching", "no_passage"),
                                ("mcq", "mcq_no_passage")):
            if full_a in table[model] and blind_a in table[model]:
                f, b = table[model][full_a][0], table[model][blind_a][0]
                if b < 1:
                    print(f"  {model:30s} {full_a:14s} {(f-b)/(1-b):.3f}")

    # ---- rankings ---------------------------------------------------------
    for mcq_a, match_a in (("mcq", "matching+d"), ("mcq", "matching")):
        models = [m for m in sorted(table) if mcq_a in table[m] and match_a in table[m]]
        if len(models) < 3:
            continue
        mcq_s = [table[m][mcq_a][0] for m in models]
        mat_s = [table[m][match_a][0] for m in models]
        print(f"\nRANKING: {mcq_a} vs {match_a}   ({len(models)} models)")
        by_mcq = sorted(models, key=lambda m: -table[m][mcq_a][0])
        by_mat = sorted(models, key=lambda m: -table[m][match_a][0])
        for i, (a, b) in enumerate(zip(by_mcq, by_mat), 1):
            flag = "" if a == b else "   <-- ORDER CHANGED"
            print(f"  {i}. {a:30s} | {b:30s}{flag}")
        rho = spearman(mcq_s, mat_s)
        print(f"  Spearman rho = {rho:.3f}"
              f"   ({'monotone - formats agree on ordering' if rho > 0.99 else 'ORDERING DIFFERS - the headline claim'})")

    # ---- paired bootstrap -------------------------------------------------
    print("\npaired bootstrap, matching+d vs mcq (resampling groups)")
    for model in sorted(table):
        if "matching+d" in groups[model] and "mcq" in groups[model]:
            r = paired_bootstrap(groups[model]["matching+d"], groups[model]["mcq"])
            if r:
                d, lo, hi, k = r
                print(f"  {model:30s} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (k={k} groups)")

    # ---- model vs model on ACCURACY ---------------------------------------
    # The counterpart to the false-match bootstrap. The claim "these two models
    # are indistinguishable on accuracy but differ hugely on abstention" needs
    # a CI on BOTH halves -- one that straddles zero and one that does not --
    # otherwise "indistinguishable" is just an eyeballed small difference.
    for arm in sorted({a for m in groups.values() for a in m}):
        models = sorted(m for m in groups if arm in groups[m] and groups[m][arm])
        if len(models) < 2:
            continue
        print(f"\npaired bootstrap on ACCURACY between models, {arm}")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                r = paired_bootstrap(groups[models[i]][arm], groups[models[j]][arm])
                if not r:
                    continue
                d, lo, hi, k = r
                tag = "" if lo <= 0 <= hi else "   SIGNIFICANT"
                print(f"  {models[i]:26s} - {models[j]:26s} {d:+.4f}  "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}]  (k={k}){tag}")


if __name__ == "__main__":
    main()
