#!/usr/bin/env python
# ============================================================================
# position_bias.py — does mcq_none's abstention measurement depend on WHERE the
# "none of these" option happened to land?
#
# The claim under test: offering abstention as a lettered option entangles it
# with option-ID selection bias (Zheng et al., ICLR'24 -- models hold a priori
# preferences over A/B/C/D). If so, a model abstains more when "none" sits on
# its preferred letter, and the measured false-match rate is partly an artifact
# of a coin flip made at build time. Matching cannot have this problem: "none"
# is a token the model emits, not a position it selects.
#
# Two tests, both on withheld-gold items where abstaining is CORRECT:
#   1. abstention rate by the letter "none" occupied
#   2. permutation test on the spread (max-min), so we do not read noise as bias
# Plus the mirror image on answerable items: wrongly picking "none" by position.
#
# A NULL RESULT HERE ARGUES AGAINST US and should be reported as such.
#
#   python position_bias.py --src h_mcqnone40.jsonl --glob "h_mcqnone40.*.results.jsonl"
# ============================================================================
import argparse
import glob
import json
import os
import random
from collections import defaultdict


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


def perm_p(labels, hits, iters=20000, seed=0):
    """P(spread across positions >= observed) when position is irrelevant.

    Shuffling the position labels against fixed outcomes is the exact null:
    it destroys any position-outcome link while preserving both the number of
    items per position and the overall abstention rate.
    """
    def spread(lab):
        agg = defaultdict(lambda: [0, 0])
        for L, h in zip(lab, hits):
            agg[L][0] += h
            agg[L][1] += 1
        rates = [ok / n for ok, n in agg.values() if n >= 10]
        return (max(rates) - min(rates)) if len(rates) > 1 else 0.0

    obs = spread(labels)
    rng = random.Random(seed)
    shuf = list(labels)
    ge = 0
    for _ in range(iters):
        rng.shuffle(shuf)
        ge += spread(shuf) >= obs
    return obs, (ge + 1) / (iters + 1)


def trend_p(labels, hits, iters=20000, seed=0):
    """Permutation test on the A->D TREND, not the unordered spread.

    The observed effect is monotone in position for most models, and max-minus-min
    is blind to ordering -- it scores A>B>C>D and a random jumble alike, so it is
    badly underpowered here. Correlating position index with outcome uses the
    ordering and is the right test for 'models drift toward later letters'.
    """
    idx = [ord(L) - 65 for L in labels]
    n = len(idx)
    mi = sum(idx) / n
    mh = sum(hits) / n
    dv = sum((x - mi) ** 2 for x in idx) ** 0.5 * sum((h - mh) ** 2 for h in hits) ** 0.5
    if not dv:
        return 0.0, 1.0
    obs = sum((x - mi) * (h - mh) for x, h in zip(idx, hits)) / dv
    rng = random.Random(seed)
    shuf = list(idx)
    ge = 0
    for _ in range(iters):
        rng.shuffle(shuf)
        r = sum((x - mi) * (h - mh) for x, h in zip(shuf, hits)) / dv
        ge += abs(r) >= abs(obs)
    return obs, (ge + 1) / (iters + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="h_mcqnone40.jsonl")
    ap.add_argument("--glob", default="h_mcqnone40.*.results.jsonl")
    a = ap.parse_args()

    # results files predate none_letter being recorded -> join back to source
    meta = {}
    for r in load(a.src):
        k = r.get("item_id") or r.get("group_id")
        meta[k] = (r.get("none_letter", ""), bool(r.get("withheld")))
    if not meta:
        raise SystemExit(f"no records in {a.src}")
    print(f"source: {len(meta)} items, "
          f"{sum(1 for _, w in meta.values() if w)} with the gold withheld")

    for path in sorted(glob.glob(a.glob)):
        model = os.path.basename(path).replace(".results.jsonl", "").split(".")[-1]
        # withheld items: abstaining is CORRECT
        w_lab, w_hit = [], []
        # answerable items: picking "none" is WRONG (over-abstention)
        a_lab, a_hit = [], []
        for r in load(path):
            k = r.get("key") or r.get("item_id") or r.get("group_id")
            if k not in meta:
                continue
            nl, withheld = meta[k]
            p = r["pred"][0] if r.get("pred") else None
            if p is None or not nl:
                continue
            if withheld:
                w_lab.append(nl); w_hit.append(p == "none")
            else:
                a_lab.append(nl); a_hit.append(p == "none")
        if not w_lab:
            continue

        print(f"\n{'='*74}\n{model}")
        for name, lab, hit in (("ABSTAINED (correct) on withheld items", w_lab, w_hit),
                               ("wrongly abstained on answerable items", a_lab, a_hit)):
            agg = defaultdict(lambda: [0, 0])
            for L, h in zip(lab, hit):
                agg[L][0] += h
                agg[L][1] += 1
            if not agg:
                continue
            print(f"  {name}")
            for L in sorted(agg):
                ok, n = agg[L]
                bar = "#" * int(30 * ok / n) if n else ""
                print(f"    none at {L}: {ok/n:.3f}  (n={n:4d})  {bar}")
            obs, p = perm_p(lab, hit)
            print(f"    spread {obs:.3f}, permutation p={p:.4f}"
                  f"{'  SIG' if p < 0.05 else ''}")
            r, pt = trend_p(lab, hit)
            verdict = ("POSITION MATTERS — abstention drifts with letter position"
                       if pt < 0.05 else
                       "no trend — argues AGAINST our structural claim")
            print(f"    A->D trend r={r:+.4f}, permutation p={pt:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
