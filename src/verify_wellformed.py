#!/usr/bin/env python
# ============================================================================
# verify_wellformed.py — the W1/W2 verifier screen (and W4 by corollary).
#
# For every question in every group, a screener model sees the passage, the
# question, the FULL candidate pool, and the gold REVEALED, and answers one
# binary: is any OTHER candidate also a fully correct answer? Verification
# with the gold revealed is far easier than solving, which is what makes a
# cheap screener trustworthy here.
#
# A question is FLAGGED when >= --votes screeners name another candidate.
# A group is flagged when any of its questions is. Two violation types are
# distinguished, because they mean different things:
#   * the named candidate is another question's GOLD  -> W1/W2 violation
#     (semantic duplicate golds or cross-valid gold)
#   * the named candidate is a distractor             -> cross-valid
#     distractor (harmless for its own question, wrong-by-authorship only
#     there)
# Both make the flagged QUESTION ambiguous, so both flag the group.
#
# W4 corollary: a withheld question stays sound exactly when no remaining
# candidate is defensible for it, which is what this screen certifies on the
# full pool. No separate withhold screen is needed.
#
# Outputs:
#   flagged_groups.txt          group_ids, feed to compare.py --exclude-groups
#   wellformed_detail.jsonl     per-question verdicts, for the paper appendix
#
#   python verify_wellformed.py --src h_match.jsonl --limit 300
#   python compare.py --glob "h_*.results.jsonl" --exclude-groups flagged_groups.txt
# ============================================================================
import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from run_matching import LETTERS, make_client, call

PROMPT = """You are auditing a benchmark item for ambiguity.

PASSAGE:
{passage}

QUESTION: {question}

CANDIDATE ANSWERS:
{block}

The intended correct answer is {gold}.

Based only on the passage, is any OTHER candidate ALSO a fully correct answer
to this question? A candidate that is merely related or partially true does
not count. It must be a fully correct answer on its own.

Reply with exactly one token: the letter of such a candidate, or NO."""


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


def verdict(text, gold_letter, m):
    """-> flagged letter, 'NO', or None (unparsable)."""
    if not text:
        return None
    u = text.strip().upper()
    if re.search(r"\bNO\b", u):
        return "NO"
    mt = re.search(r"\b([A-Z])\b", u)
    if mt and mt.group(1) in set(LETTERS[:m]) and mt.group(1) != gold_letter:
        return mt.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="h_match.jsonl",
                    help="a FULL-POOL matching file (no withholding)")
    ap.add_argument("--models",
                    default="gemini-3.6-flash,gemini-3.1-flash-lite,"
                            "gemma-4-26b-a4b-it-maas")
    ap.add_argument("--votes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="groups, 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=0)
    ap.add_argument("--max-out", type=int, default=16)
    ap.add_argument("--out", default="flagged_groups.txt")
    ap.add_argument("--detail", default="wellformed_detail.jsonl")
    a = ap.parse_args()

    recs = load(a.src)
    if a.limit:
        recs = recs[:a.limit]
    if not recs or recs[0].get("arm") != "matching" or recs[0].get("withheld"):
        raise SystemExit("--src must be a full-pool matching file (h_match.jsonl)")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    print(f"{len(recs)} groups x {recs[0]['n']} questions x {len(models)} screeners "
          f"= {len(recs)*recs[0]['n']*len(models)} calls")

    client = make_client()
    jobs = []                                   # (gid, qi, model, prompt, gold_L, m)
    for r in recs:
        cands = r["candidates"]
        block = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cands))
        # the passage is inside r["prompt"] between fixed markers (build_prompt:
        # "PASSAGE:\n...\n\nQUESTIONS:"). Fall back to the whole prompt if the
        # markers ever change -- a screener with extra text beats one with none.
        pr = r["prompt"]
        if "PASSAGE:" in pr and "\n\nQUESTIONS:" in pr:
            passage = pr.split("PASSAGE:", 1)[1].split("\n\nQUESTIONS:", 1)[0].strip()
        else:
            # knowledge benchmarks (MMLU-Pro, GPQA, C-Eval) have no passage:
            # tell the screener to judge from expert knowledge instead of
            # feeding it the whole prompt as a fake passage
            passage = ("(This benchmark has no passage. Judge from expert "
                       "knowledge of the subject.)")
        for qi, (q, gold_L) in enumerate(zip(r["questions"], r["answer"])):
            p = PROMPT.format(passage=passage, question=q, block=block, gold=gold_L)
            for mdl in models:
                jobs.append((r["group_id"], qi, mdl, p, gold_L, len(cands)))

    def work(job):
        gid, qi, mdl, p, gold_L, m = job
        txt, err = call(client, mdl, p, a.max_out, a.rpm)
        return (gid, qi, mdl, verdict(txt, gold_L, m), err)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(work, jobs))

    votes = defaultdict(list)                   # (gid, qi) -> [flagged letters]
    unparsed = errs = 0
    for gid, qi, mdl, v, err in results:
        if err:
            errs += 1
        if v is None:
            unparsed += 1
        elif v != "NO":
            votes[(gid, qi)].append(v)

    by_gid = {r["group_id"]: r for r in recs}
    flagged_q, flagged_groups = [], set()
    n_gold_type = n_distractor_type = 0
    for (gid, qi), letters in votes.items():
        if len(letters) < a.votes:
            continue
        r = by_gid[gid]
        golds = set(r["answer"])
        named = max(set(letters), key=letters.count)
        typ = "gold" if named in golds else "distractor"
        if typ == "gold":
            n_gold_type += 1
        else:
            n_distractor_type += 1
        flagged_q.append(dict(group_id=gid, q_index=qi, named=named, type=typ,
                              votes=len(letters)))
        flagged_groups.add(gid)

    total_q = len(recs) * recs[0]["n"]
    print(f"\nscreeners disagree-with-uniqueness on {len(flagged_q)}/{total_q} "
          f"questions ({100*len(flagged_q)/total_q:.1f}%)")
    print(f"  named candidate is another question's gold : {n_gold_type}  (W1/W2)")
    print(f"  named candidate is a distractor            : {n_distractor_type}  (cross-valid)")
    print(f"flagged groups: {len(flagged_groups)}/{len(recs)}  ->  RETENTION "
          f"{100*(1-len(flagged_groups)/len(recs)):.1f}%")
    print(f"unparsed screener replies {unparsed}, api errors {errs} "
          f"(both counted as clean, reported here)")

    with open(a.out, "w", encoding="utf-8") as f:
        for gid in sorted(flagged_groups):
            f.write(gid + "\n")
    with open(a.detail, "w", encoding="utf-8") as f:
        for d in flagged_q:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n-> {a.out}  (feed to: python compare.py --glob \"h_*.results.jsonl\" "
          f"--exclude-groups {a.out})")
    print(f"-> {a.detail}")


if __name__ == "__main__":
    main()
