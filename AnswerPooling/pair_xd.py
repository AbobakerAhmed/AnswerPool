#!/usr/bin/env python
# ============================================================================
# pair_xd.py -- the cross-domain ablation on identical questions.
#
# collect_bench.py scores a cross-domain file on all of its groups: result
# records do not carry source_groups, so its pairing filter passes every
# record. A regrouped build also leaves out the questions it cannot place in
# a group of N distinct contexts (GPQA keeps 67 of 83 groups). Its MCQ,
# pooled and cross-domain columns were therefore not on the same questions.
#
# This script joins every cross-domain slot back to its source group and slot
# through the two build files, and scores MCQ, pooled and cross-domain on the
# questions that every model with all three files answered, inside the groups
# shared by every pooled and MCQ file (the paired set of the paper).
#
#   python -m AnswerPooling.pair_xd --prefix ra
#   python -m AnswerPooling.pair_xd --prefix gp
#   python -m AnswerPooling.pair_xd --prefix h
# ============================================================================
import argparse
import glob
import re

from .build_matching import qkey
from .collect_bench import split_name
from .collect_quality import load

LOCAL5 = "Qwen3-32B,Gemma 3 27B,Gemma 3 12B,Qwen3-8B,Qwen2.5-7B"


def xd_slots(prefix):
    """(cross-domain group id, slot) -> (source group id, source slot)."""
    src = {r["group_id"]: {qkey(q): i for i, q in enumerate(r["questions"])}
           for r in load(f"{prefix}_match.jsonl")}
    out, missing = {}, 0
    for r in load(f"{prefix}_xd.jsonl"):
        for i, q in enumerate(r["questions"]):
            # passage benchmarks prefix each question with its passage label
            k = qkey(re.sub(r"^\[Passage \d+\]\s*", "", q))
            hit = [(g, src[g][k]) for g in r.get("source_groups", [])
                   if k in src.get(g, {})]
            if hit:
                out[(r["group_id"], i)] = hit[0]
            else:
                missing += 1
    return out, missing


def per_slot(path, xd_map=None):
    """{(source group id, slot): correct} for one results file, scored as in
    collect_quality.metrics: a slot is right when the parsed letter equals the
    gold, so an unparsed slot counts as wrong."""
    out = {}
    for r in load(path):
        preds, golds = r["pred"], r["gold"]
        for i, g in enumerate(golds):
            if g == "none":
                continue
            p = preds[i] if i < len(preds) else None
            if xd_map is not None:
                key = xd_map.get((r["group_id"], i))
                if key is None:
                    continue
            elif r.get("q_index") is not None:       # MCQ: one record per question
                key = (r["group_id"], r["q_index"])
            else:
                key = (r["group_id"], i)
            out[key] = (p == g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="ra, gp, h, ...")
    ap.add_argument("--models", default=LOCAL5,
                    help="models averaged in the summary line, comma separated")
    a = ap.parse_args()
    p = a.prefix

    xd_map, missing = xd_slots(p)
    files = {role: {split_name(f)[1]: f
                    for f in glob.glob(f"{p}_{stem}.*.results.jsonl")}
             for role, stem in (("mcq", "mcq"), ("pooled", "match"), ("xd", "xd"))}

    keep = None
    for role in ("mcq", "pooled"):
        for f in files[role].values():
            g = {r["group_id"] for r in load(f)}
            keep = g if keep is None else keep & g
    if not keep:
        raise SystemExit(f"no groups shared by the {p}_mcq and {p}_match results")

    slots = {m: (per_slot(files["mcq"][m]), per_slot(files["pooled"][m]),
                 per_slot(files["xd"][m], xd_map))
             for m in sorted(files["xd"])
             if m in files["mcq"] and m in files["pooled"]}
    if not slots:
        raise SystemExit(f"no model has all three of {p}_mcq, {p}_match, {p}_xd results")
    common = None
    for mc, po, xd in slots.values():
        s = {k for k in xd if k[0] in keep and k in mc and k in po}
        common = s if common is None else common & s
    if not common:
        raise SystemExit("no question is shared by every model in all three formats")

    print(f"prefix {p}: {len(xd_map)} cross-domain slots joined to their source, "
          f"{missing} not found")
    print(f"{len(keep)} groups shared by every pooled and MCQ file; "
          f"{len(common)} questions answered by all {len(slots)} models in all three formats\n")
    print(f"{'model':16s} {'MCQ':>7s} {'pooled':>7s} {'cross':>7s} {'recovered':>10s}")
    rows = {}
    for m, dicts in slots.items():
        acc = [sum(d[k] for k in common) / len(common) for d in dicts]
        rows[m] = acc
        gap = acc[0] - acc[1]
        rec = f"{100 * (acc[2] - acc[1]) / gap:9.1f}%" if gap else "      --"
        print(f"{m:16s} {acc[0]:7.3f} {acc[1]:7.3f} {acc[2]:7.3f} {rec}")

    sel = [m for m in a.models.split(",") if m in rows]
    if sel:
        mean = [sum(rows[m][i] for m in sel) / len(sel) for i in range(3)]
        gap = mean[0] - mean[1]
        above = sum(rows[m][2] > rows[m][1] for m in sel)
        print(f"\nmean of {len(sel)} ({', '.join(sel)}):")
        print(f"  MCQ {mean[0]:.3f}  pooled {mean[1]:.3f}  cross-domain {mean[2]:.3f}  "
              f"recovered {100 * (mean[2] - mean[1]) / gap:.1f}% of the drop, "
              f"cross-domain above pooled for {above} of {len(sel)}")


if __name__ == "__main__":
    main()
