#!/usr/bin/env python
# ============================================================================
# collect_bench.py -- one benchmark, every model, in the paper's table shape.
#
#   python -m AnswerPooling.collect_bench --prefix mp9          # MMLU-Pro, 9 of 9 build
#   python -m AnswerPooling.collect_bench --prefix ra | tee results_ra.txt
#
# Reads <prefix>_*.results.jsonl, classifies each file by the arm its records
# carry, restricts every number to the group ids shared by every model's
# pooled and MCQ files (so the columns are item-paired), and prints:
#   HEALTH      unparsed / errors per file, groups per stem
#   TABLE       per model: MCQ / pooled accuracy, MCQ / pooled all-correct,
#               options-only accuracy and leakage alpha, unrelated pool,
#               cross-domain pool (when a <prefix>_xd file exists)
#   FOOTER      N, M, k, both floors, models harder, rank correlation rho,
#               MCQ minus cross-domain accuracy, mean over models
#   REMOVED     per model: false answer under pooled / MCQ+instruction /
#               MCQ+none, false abstention under pooled / MCQ+none
# Point estimates only. Nothing here is a confidence interval.
# ============================================================================
import argparse
import glob
import os
from collections import defaultdict

from .collect_quality import load, metrics, fmt

# arm field of the record -> role in the paper
ROLE = {
    ("matching", False): "pooled",
    ("matching", True): "pooled_rm",
    ("mcq", False): "mcq",
    ("mcq_none", True): "mcq_none",
    ("mcq_prose", True): "mcq_instr",
    ("choices_only", False): "options_only",
    ("easy", False): "unrelated",
    ("mcq_no_passage", False): "mcq_nocontext",
    ("no_passage", False): "pooled_nocontext",
    ("mcq_choices_only", False): "mcq_options_only",
}

SHORT = {
    "Qwen_Qwen3-32B-FP8": "Qwen3-32B", "Qwen_Qwen3-32B": "Qwen3-32B",
    "Qwen_Qwen3-8B": "Qwen3-8B", "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "google_gemma-3-27b-it": "Gemma 3 27B", "google_gemma-3-12b-it": "Gemma 3 12B",
    "gemini-3.6-flash": "Gemini 3.6 Flash", "gemini-3.1-flash-lite": "Gemini 3.1 F-Lite",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemma-4-26b-a4b-it-maas": "Gemma 4 26B",
    "google_gemma-4-26b-a4b-it": "Gemma 4 26B",   # local vLLM run, same weights
    "openai_gpt-oss-120b": "gpt-oss-120b", "openai_gpt-oss-20b": "gpt-oss-20b",
    "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B", "Qwen_Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen_Qwen3-VL-32B-Instruct-FP8": "Qwen3-VL-32B",
    "OpenGVLab_InternVL3-8B-hf": "InternVL3-8B", "OpenGVLab_InternVL3-38B-hf": "InternVL3-38B",
    "OpenGVLab_InternVL3-14B-hf": "InternVL3-14B", "OpenGVLab_InternVL3-14B": "InternVL3-14B",
    "Qwen_Qwen3.5-27B": "Qwen3.5-27B", "Qwen_Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen_Qwen3.5-9B": "Qwen3.5-9B", "Qwen_Qwen3.5-35B-A3B": "Qwen3.5-35B-A3B",
    "OpenGVLab_InternVL3-8B": "InternVL3-8B", "OpenGVLab_InternVL3-38B": "InternVL3-38B",
}


def split_name(path):
    """<stem>.<model>.results.jsonl -> (stem, model). Stems carry no dots."""
    base = os.path.basename(path)[: -len(".results.jsonl")]
    stem, _, model = base.partition(".")
    return stem, SHORT.get(model, model)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def perm_floor(m, n):
    p = 1.0
    for i in range(n):
        p /= (m - i)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="file prefix, e.g. mp9, ra, gp, ce, mm, mx")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--xd", default="", help="stem of the cross-domain pooled file, "
                                             "default <prefix>_xd")
    a = ap.parse_args()
    xd_stem = a.xd or f"{a.prefix}_xd"

    files = sorted(glob.glob(os.path.join(a.dir, f"{a.prefix}_*.results.jsonl")))
    files = [f for f in files if ".judged." not in f]
    if not files:
        raise SystemExit(f"no {a.prefix}_*.results.jsonl in {a.dir}")

    RAW = defaultdict(dict)       # RAW[role][model] = records
    stems = defaultdict(set)
    owner = {}                    # (role, model) -> stem that holds it
    # the stems README.md builds. When an older file plays the same role for a
    # model (a no-context screen under another name), the standard stem wins.
    canon = {f"{a.prefix}_{s}" for s in (
        "match", "mcq", "np", "mcq_np", "noimg", "easy", "wh40", "wh50",
        "mcqnone40", "prose40", "mcq_none", "mcq_prose", "co", "xd")}
    n_q = m_pool = k_opt = None
    for f in files:
        stem, model = split_name(f)
        recs = load(f)
        if not recs:
            continue
        r0 = recs[0]
        removed = any(r.get("withheld") for r in recs[:50])
        role = ROLE.get((r0.get("arm", ""), removed))
        if stem == xd_stem:
            role = "xd"
        # an mcq arm built with --drop-images (mm_noimg) or a *_np stem is the
        # no-context probe, not the MCQ baseline; likewise a pooled file built
        # with --drop-images for video
        if role == "mcq" and (stem.endswith("_noimg") or stem.endswith("_np")):
            role = "mcq_nocontext"
        if role == "pooled" and (stem.endswith("_noimg") or stem.endswith("_np")):
            role = "pooled_nocontext"
        if r0.get("cross_domain"):
            role = "xd_rm" if removed else "xd"
        if role is None:
            print(f"  skipping {os.path.basename(f)}: arm={r0.get('arm')} removed={removed}")
            continue
        if role in ("pooled", "xd") and r0.get("n"):
            n_q, m_pool = r0["n"], r0["m"]
        if role == "mcq" and r0.get("m"):
            k_opt = r0["m"]
        if model in RAW[role]:
            prev = owner[(role, model)]
            if stem in canon and prev not in canon:
                print(f"  note: {role} for {model} taken from {stem}, not {prev}")
            else:
                print(f"  WARNING: two files play the role {role} for {model} "
                      f"(stems {prev}/{stem}); keeping {prev}")
                continue
        stems[role].add(stem)
        owner[(role, model)] = stem
        RAW[role][model] = recs

    print(f"prefix {a.prefix}: roles found " +
          ", ".join(f"{r}[{len(RAW[r])} models, stem {'/'.join(sorted(stems[r]))}]"
                    for r in RAW))

    # ---- health ------------------------------------------------------------
    print("\n" + "=" * 78 + "\nHEALTH  (unparsed and errors must be 0)\n" + "=" * 78)
    for role in RAW:
        for model, recs in RAW[role].items():
            x = metrics(recs)
            gcount = x["groups"]
            flag = "" if not (x["unparsed"] or x["errors"]) else "   <-- check"
            print(f"  {role:16s} {model:20s} groups={gcount:5d} unparsed={x['unparsed']:4d} "
                  f"errors={x['errors']:4d}{flag}")

    # ---- paired group set: shared by every pooled and every mcq file --------
    sets = [{r["group_id"] for r in recs} for role in ("pooled", "mcq")
            for recs in RAW[role].values()]
    keep = set.intersection(*sets) if sets else None
    if keep is not None:
        print(f"\nall numbers below are on the {len(keep)} groups shared by every "
              f"pooled and MCQ file")

    def M(role, model, keep_set=None):
        recs = RAW[role].get(model)
        if not recs:
            return None
        ks = keep if keep_set is None else keep_set
        if role in ("xd", "xd_rm"):
            # cross-domain groups have their own ids; pair them through the
            # same-context groups their questions came from
            if ks is not None:
                recs = [r for r in recs if all(s in ks for s in r.get("source_groups", []))]
            return metrics(recs) if recs else None
        return metrics(recs, ks)

    models = sorted(set(RAW["pooled"]) | set(RAW["mcq"]),
                    key=lambda m: -(M("mcq", m) or {"acc": -1})["acc"])

    # ---- main table --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"TABLE  accuracy MCQ / pooled | all-correct MCQ / pooled | options only, alpha | "
          f"unrelated | cross-domain")
    print("=" * 78)
    cap = 1 / n_q if n_q else None
    floor_acc = 1 / m_pool if m_pool else None
    xs, ys, harder = [], [], 0
    fcost = []
    for m in models:
        mc, po = M("mcq", m), M("pooled", m)
        co, ea, xd = M("options_only", m), M("unrelated", m), M("xd", m)
        alpha = None
        if co and cap and floor_acc:
            alpha = max(0.0, (co["acc"] - floor_acc) / (cap - floor_acc))
        if mc and po:
            xs.append(mc["acc"]); ys.append(po["acc"]); harder += po["acc"] < mc["acc"]
        if mc and xd:
            fcost.append(mc["acc"] - xd["acc"])
        print(f"  {m:20s} {fmt(mc['acc']) if mc else '  --  '} / {fmt(po['acc']) if po else '  --  '}"
              f"   |   {fmt(mc['allc']) if mc else '  --  '} / {fmt(po['allc']) if po else '  --  '}"
              f"   |   {fmt(co['acc']) if co else '  --  '}, {fmt(alpha, 2) if alpha is not None else ' -- '}"
              f"   |   {fmt(ea['acc']) if ea else '  --  '}"
              f"   |   {fmt(xd['acc']) if xd else '  --  '}")

    print("\n  footer")
    if n_q and m_pool:
        print(f"    N = {n_q}, M = {m_pool}, k = {k_opt}")
        if k_opt:
            print(f"    floor, accuracy     {1/k_opt:.3f} / {1/m_pool:.3f}")
            print(f"    floor, all-correct  {k_opt**-n_q:.2e} / {perm_floor(m_pool, n_q):.2e}")
    print(f"    models harder       {harder}/{len(xs)}")
    print(f"    rank correlation    {spearman(xs, ys):.2f}  over {len(xs)} models")
    if fcost:
        gap = (1 / k_opt - 1 / m_pool) if (k_opt and m_pool) else None
        print(f"    MCQ - cross-domain  {sum(fcost)/len(fcost):+.3f}  mean over {len(fcost)} models "
              f"(range {min(fcost):+.3f} to {max(fcost):+.3f})"
              + (f"; floor gap 1/k - 1/M = {gap:.3f}, what a pure guess loses" if gap else ""))
    else:
        print(f"    MCQ - cross-domain  -- (no {xd_stem} file yet)")

    # ---- removed mode ------------------------------------------------------
    if RAW["pooled_rm"]:
        rm_sets = [{r["group_id"] for r in recs} for recs in RAW["pooled_rm"].values()]
        keep_rm = set.intersection(*rm_sets)
        print("\n" + "=" * 78)
        print(f"REMOVED  false answer pooled / cross-domain / MCQ+instruction / MCQ+none | "
              f"false abstention pooled / MCQ+none | acc pooled | all-correct pooled, "
              f"abstentions included   ({len(keep_rm)} shared groups)")
        print("=" * 78)
        for m in sorted(RAW["pooled_rm"]):
            def R(role):
                return M(role, m, keep_rm)
            p, x, i, nn = R("pooled_rm"), R("xd_rm"), R("mcq_instr"), R("mcq_none")
            print(f"  {m:20s} {fmt(p['fa'])} / {fmt(x['fa']) if x else '  --  '} / "
                  f"{fmt(i['fa']) if i else '  --  '} / {fmt(nn['fa']) if nn else '  --  '}"
                  f"   |   {fmt(p['fab'])} / {fmt(nn['fab']) if nn else '  --  '}   |   {fmt(p['acc'])}"
                  f"   |   {fmt(p['allc'])}")

    # ---- no-context, when present ------------------------------------------
    if RAW["mcq_nocontext"] or RAW["pooled_nocontext"]:
        print("\n" + "=" * 78 + "\nNO CONTEXT  accuracy with the context removed, MCQ / pooled, "
              "beside the with-context numbers\n" + "=" * 78)
        for m in sorted(set(RAW["mcq_nocontext"]) | set(RAW["pooled_nocontext"])):
            a, b = M("mcq_nocontext", m), M("pooled_nocontext", m)
            mc, po = M("mcq", m), M("pooled", m)
            print(f"  {m:20s} {fmt(a['acc']) if a else '  --  '} / {fmt(b['acc']) if b else '  --  '}"
                  f"   (with context {fmt(mc['acc']) if mc else '  --  '} / {fmt(po['acc']) if po else '  --  '})")


if __name__ == "__main__":
    main()
