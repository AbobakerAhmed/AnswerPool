#!/usr/bin/env python
# ============================================================================
# diagnose.py — two checks that must pass before trusting any run.
#
# (1) GOLD CONVENTION. Is QuALITY's `answer` field 0-based or 1-based?
#     If gold_of() is off by one, every gold is the PREVIOUS option: MCQ
#     accuracy collapses to ~0 (observed 0.045 vs 0.25 chance) while matching
#     survives, because a wrong option is still topically tied to its own
#     question. Resolved here by asking a model the raw MCQ and seeing which
#     convention its answers agree with.
#
# (2) LEXICAL BASELINE. Can TF-IDF alone solve the matching task? If word
#     overlap between question and candidate gets high accuracy, the task is
#     lexical association, not comprehension -- a reviewer will ask this.
#     No API calls.
#
#   python diagnose.py --check gold      (needs VERTEX_KEY / GOOGLE_API_KEY)
#   python diagnose.py --check lexical --in groups_n5.jsonl
# ============================================================================
import argparse
import json
import re
from collections import Counter


def norm(s):
    return " ".join(str(s).split()).strip()


# ------------------------------------------------------------ (1) gold ------
def check_gold(n=40):
    from datasets import load_dataset
    ds = load_dataset("emozilla/quality", split="validation")

    print("raw `answer` values seen:", Counter(str(r["answer"]) for r in list(ds)[:500]).most_common())
    print("n_options distribution :", Counter(len(r["options"]) for r in list(ds)[:500]).most_common())
    print()

    from run_matching import make_client, make_config
    client = make_client()
    rows = [ds[i] for i in range(0, len(ds), max(1, len(ds) // n))][:n]

    agree0 = agree1 = parsed = 0
    for r in rows:
        opts = [norm(o) for o in r["options"]]
        blk = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(opts))
        prompt = (f"Read the passage and answer the question.\n\nPASSAGE:\n{norm(r['article'])}\n\n"
                  f"QUESTION: {norm(r['question'])}\n\nOPTIONS:\n{blk}\n\n"
                  f"Respond with only the letter of the correct option.")
        try:
            resp = client.models.generate_content(
                model="gemini-3.1-flash-lite", contents=[prompt],
                config=make_config(64)).text or ""
        except Exception as e:
            print(f"  call failed: {type(e).__name__}"); continue
        m = re.search(r"\b([A-D])\b", resp.strip().upper())
        if not m:
            continue
        parsed += 1
        pick = ord(m.group(1)) - 65                 # 0-based index the model chose
        a = r["answer"]
        a = int(a) if str(a).isdigit() else (ord(str(a).upper()[0]) - 65)
        agree0 += (pick == a)                       # answer is 0-BASED
        agree1 += (pick == a - 1)                   # answer is 1-BASED

    print(f"parsed {parsed}/{len(rows)} responses")
    print(f"  model agrees with 0-based reading : {agree0}/{parsed} = {agree0/max(1,parsed):.2f}")
    print(f"  model agrees with 1-based reading : {agree1}/{parsed} = {agree1/max(1,parsed):.2f}")
    print()
    if agree0 > agree1 * 1.5:
        print(">>> `answer` is 0-BASED. gold_of() must use opts[a], NOT opts[a-1]. FIX AND REBUILD.")
    elif agree1 > agree0 * 1.5:
        print(">>> `answer` is 1-BASED. gold_of() is correct; the MCQ collapse has another cause.")
    else:
        print(">>> inconclusive - inspect a few items by hand.")


# --------------------------------------------------------- (2) lexical -----
_WORD = re.compile(r"[a-z0-9']+")


def toks(s):
    return _WORD.findall(s.lower())


def check_lexical(path, limit=200):
    import math
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()][:limit]
    recs = [r for r in recs if r["arm"] == "matching"]
    if not recs:
        print("no matching-arm records"); return

    # idf over all candidates in the file
    df = Counter()
    for r in recs:
        for c in r["candidates"]:
            for w in set(toks(c)):
                df[w] += 1
    Nd = sum(len(r["candidates"]) for r in recs)
    idf = {w: math.log(1 + Nd / (1 + d)) for w, d in df.items()}

    def score(q, c):
        qs, cs = set(toks(q)), set(toks(c))
        return sum(idf.get(w, 0.0) for w in qs & cs)

    ok = tot = 0
    exact_ok = 0
    for r in recs:
        # greedy injective assignment by descending score
        pairs = sorted(((score(q, c), qi, ci)
                        for qi, q in enumerate(r["questions"])
                        for ci, c in enumerate(r["candidates"])), reverse=True)
        used_q, used_c, assign = set(), set(), {}
        for s, qi, ci in pairs:
            if qi in used_q or ci in used_c:
                continue
            assign[qi] = ci; used_q.add(qi); used_c.add(ci)
        all_ok = True
        for qi, g in enumerate(r["answer"]):
            if g == "none":
                continue
            tot += 1
            got = chr(65 + assign.get(qi, -1)) if qi in assign else None
            if got == g:
                ok += 1
            else:
                all_ok = False
        exact_ok += all_ok

    M = recs[0]["m"]
    print(f"TF-IDF greedy assignment on {len(recs)} groups")
    print(f"  per-pair accuracy {ok}/{tot} = {ok/max(1,tot):.4f}   (chance {1/M:.4f})")
    print(f"  exact assignment  {exact_ok}/{len(recs)} = {exact_ok/len(recs):.4f}")
    print()
    if ok / max(1, tot) > 0.35:
        print(">>> WARNING: lexical overlap solves a large share. The task may be "
              "measuring word association, not comprehension. Report this baseline "
              "and consider filtering high-overlap groups.")
    else:
        print(">>> lexical baseline is weak - good, comprehension is doing the work.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["gold", "lexical"], required=True)
    ap.add_argument("--in", dest="inp", default="groups_n5.jsonl")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.check == "gold":
        check_gold(a.n)
    else:
        check_lexical(a.inp)
