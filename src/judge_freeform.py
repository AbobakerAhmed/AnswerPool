#!/usr/bin/env python
# ============================================================================
# judge_freeform.py — classify open-ended responses as ANSWER or ABSTAIN, so the
# freeform arm can act as the external criterion that arbitrates between
# matching and mcq_none.
#
# The judge sees ONLY the question and the response. Not the passage, not the
# gold. We are not asking "is this answer correct" -- that would need the
# passage and would confound answer quality with the willingness to answer at
# all. We are asking one binary: did the model produce a substantive answer, or
# did it decline. That is the behaviour the closed-form false-match rates claim
# to predict, and keeping the judge's job this narrow is what makes it reliable
# enough to be a criterion.
#
# Writes <stem>.judged.results.jsonl with pred = "none" (abstained) or
# "answered", so compare.py scores false-match/miss with no changes.
#
#   python judge_freeform.py --glob "h_free40.*.results.jsonl" --judge gemini-3.6-flash
# ============================================================================
import argparse
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from run_matching import make_client, call

PROMPT = """You are classifying how a language model responded to a question.

QUESTION:
{q}

RESPONSE:
{a}

Did the response provide a substantive answer to the question, or did it decline \
-- for example by saying the passage does not contain the answer, that it cannot \
tell, or that the question is not addressed by the text?

A response that gives an answer and also hedges still counts as ANSWER.
A response that only explains why it cannot answer counts as ABSTAIN.

Reply with exactly one word: ANSWER or ABSTAIN."""


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


def verdict(text):
    if not text:
        return None
    u = text.strip().upper()
    if re.search(r"\bABSTAIN\b", u):
        return "none"
    if re.search(r"\bANSWER\b", u):
        return "answered"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="h_free40.*.results.jsonl")
    ap.add_argument("--judge", default="gemini-3.6-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=0)
    a = ap.parse_args()

    client = make_client()
    for path in sorted(glob.glob(a.glob)):
        if ".judged." in path:
            continue
        recs = load(path)
        if not recs or recs[0].get("arm") != "freeform":
            continue
        model = os.path.basename(path).replace(".results.jsonl", "").split(".")[-1]

        def one(r):
            q = (r.get("questions") or [""])[0]
            txt, err = call(client, a.judge,
                            PROMPT.format(q=q, a=(r.get("raw") or "")[:3000]),
                            64, a.rpm)
            return verdict(txt)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            verdicts = list(ex.map(one, recs))

        fm = fmt = miss = pres = unj = 0
        for r, v in zip(recs, verdicts):
            if v is None:
                unj += 1
            r["pred"] = [v]
            if v is None:
                continue
            if r.get("withheld"):
                fmt += 1
                fm += (v != "none")          # answered a question with no answer
            else:
                pres += 1
                miss += (v == "none")        # declined an answerable question

        out = path.replace(".results.jsonl", ".judged.results.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"{model:32s} confabulation {fm}/{fmt} = "
              f"{fm/fmt if fmt else 0:.4f}   over-refusal {miss}/{pres} = "
              f"{miss/pres if pres else 0:.4f}   unjudged {unj}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
