#!/usr/bin/env python
# ============================================================================
# verify_wellformed.py -- the ambiguity check.
#
# For every question in every group, a screener model sees the context, the
# question, the full pool, and the question's correct answer, and names any
# OTHER option that is also a fully correct answer, or replies NO. Verifying
# with the answer revealed is far easier than solving, which is what makes a
# cheap screener usable here.
#
# A question is FLAGGED when at least --votes screeners name another option,
# and a group is kept when none of its questions is flagged. A reply that
# cannot be parsed, and an API error, count as no flag and are reported.
# Flags are split by what the screeners named:
#   * another question's correct answer
#   * a wrong option, of this question or of another one
#
# Outputs:
#   flagged_groups.txt          ids of the groups with a flagged question
#   wellformed_detail.jsonl     per-question verdicts
#
#   python -m AnswerPooling.verify_wellformed --src h_match.jsonl --limit 300
#   python -m AnswerPooling.compare --glob "h_match.*.results.jsonl" --exclude-groups flagged_groups.txt
# ============================================================================
import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .run_matching import label, labels, label_pat, load_images

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
    """-> a flagged label, 'NO', or None (unparsable)."""
    if not text:
        return None
    u = text.strip().upper()
    if re.search(r"\bNO\b", u):
        return "NO"
    mt = re.search(rf"\b({label_pat(m)})\b", u)
    if mt and mt.group(1) in set(labels(m)) and mt.group(1) != gold_letter:
        return mt.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="h_match.jsonl",
                    help="a FULL-POOL matching file (no removed answers)")
    ap.add_argument("--models",
                    default="gemini-3.6-flash,gemini-3.1-flash-lite,"
                            "gemma-4-26b-a4b-it-maas",
                    help="comma-separated screeners. With --backend vllm these "
                         "are local checkpoints")
    ap.add_argument("--backend", default="api", choices=["api", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--max-pixels", type=int, default=0)
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

    jobs = []                                   # (gid, qi, prompt, gold_L, m, imgs)
    for r in recs:
        cands = r["candidates"]
        block = "\n".join(f"{label(i)}. {c}" for i, c in enumerate(cands))
        # the passage sits inside r["prompt"] between fixed markers (build_prompt:
        # "PASSAGE:\n...\n\nQUESTIONS:")
        pr = r["prompt"]
        if "PASSAGE:" in pr and "\n\nQUESTIONS:" in pr:
            passage = pr.split("PASSAGE:", 1)[1].split("\n\nQUESTIONS:", 1)[0].strip()
        elif "frames in order" in pr:
            passage = "(The context is the attached video frames.)"
        else:
            # benchmarks without a passage (MMLU-Pro, GPQA, C-Eval)
            passage = ("(This benchmark has no passage. Judge from expert "
                       "knowledge of the subject.)")
        # an image question cannot be screened without the picture it cites,
        # so the group's images are passed through
        img_paths = r.get("images") or []
        for qi, (q, gold_L) in enumerate(zip(r["questions"], r["answer"])):
            p = PROMPT.format(passage=passage, question=q, block=block, gold=gold_L)
            jobs.append((r["group_id"], qi, p, gold_L, len(cands), img_paths))

    # results[(gid, qi, model)] = (verdict, err)
    results = {}
    if a.backend == "api":
        from .run_matching import make_client, call
        client = make_client()

        def work(job):
            gid, qi, p, gold_L, m, img_paths, mdl = job
            imgs = load_images(dict(images=img_paths)) if img_paths else []
            txt, err = call(client, mdl, p, a.max_out, a.rpm, images=imgs)
            return (gid, qi, mdl, verdict(txt, gold_L, m), err)
        api_jobs = [j + (mdl,) for j in jobs for mdl in models]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for gid, qi, mdl, v, err in ex.map(work, api_jobs):
                results[(gid, qi, mdl)] = (v, err)
    else:
        # one vLLM engine per screener, every prompt in one batch, no grammar.
        # Raw replies are cached per (source, model), so the safe way to run
        # three screeners is one process per model and a final call with all
        # three names, which only aggregates.
        import os
        from .run_matching import run_vllm
        os.makedirs("wellformed_cache", exist_ok=True)
        stem = os.path.basename(a.src).replace(".jsonl", "")
        pseudo = [dict(prompt=p, images=imgs, arm="screen", n=1, m=m)
                  for (_, _, p, _, m, imgs) in jobs]
        for mdl in models:
            cache = os.path.join("wellformed_cache",
                                 f"{stem}.{mdl.replace('/', '_')}.jsonl")
            texts = {}
            if os.path.exists(cache):
                for line in open(cache, encoding="utf-8"):
                    if line.strip():
                        d = json.loads(line)
                        texts[(d["group_id"], d["q_index"])] = d["raw"]
                print(f"{mdl}: {len(texts)} cached replies from {cache}")
            todo = [i for i, job in enumerate(jobs) if (job[0], job[1]) not in texts]
            if todo:
                got = run_vllm([pseudo[i] for i in todo], mdl, a.max_out, tp=a.tp,
                               max_len=a.max_len, guided=False, max_pixels=a.max_pixels)
                with open(cache, "a", encoding="utf-8") as f:
                    for k, i in enumerate(todo):
                        raw = got.get(k, "")
                        texts[(jobs[i][0], jobs[i][1])] = raw
                        f.write(json.dumps(dict(group_id=jobs[i][0], q_index=jobs[i][1],
                                                raw=raw), ensure_ascii=False) + "\n")
            for gid, qi, _, gold_L, m, _ in jobs:
                results[(gid, qi, mdl)] = (verdict(texts.get((gid, qi), ""), gold_L, m), None)

    votes = defaultdict(list)                   # (gid, qi) -> [flagged labels]
    unparsed = errs = 0
    for (gid, qi, mdl), (v, err) in results.items():
        if err:
            errs += 1
        if v is None:
            unparsed += 1
        elif v != "NO":
            votes[(gid, qi)].append(v)

    by_gid = {r["group_id"]: r for r in recs}
    flagged_q, flagged_groups = [], set()
    n_gold_type = n_wrong_type = 0
    for (gid, qi), named_all in votes.items():
        if len(named_all) < a.votes:
            continue
        golds = set(by_gid[gid]["answer"])
        named = max(set(named_all), key=named_all.count)
        typ = "gold" if named in golds else "wrong_option"
        if typ == "gold":
            n_gold_type += 1
        else:
            n_wrong_type += 1
        flagged_q.append(dict(group_id=gid, q_index=qi, named=named, type=typ,
                              votes=len(named_all)))
        flagged_groups.add(gid)

    total_q = sum(len(r["questions"]) for r in recs)
    print(f"\nscreeners flag {len(flagged_q)}/{total_q} questions "
          f"({100*len(flagged_q)/total_q:.1f}%)")
    print(f"  named another question's correct answer : {n_gold_type}")
    print(f"  named a wrong option                    : {n_wrong_type}")
    print(f"flagged groups: {len(flagged_groups)}/{len(recs)}  ->  kept "
          f"{100*(1-len(flagged_groups)/len(recs)):.1f}%")
    print(f"unparsed screener replies {unparsed}, api errors {errs} "
          f"(both counted as no flag)")

    with open(a.out, "w", encoding="utf-8") as f:
        for gid in sorted(flagged_groups):
            f.write(gid + "\n")
    with open(a.detail, "w", encoding="utf-8") as f:
        for d in flagged_q:
            d["src"] = a.src
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n-> {a.out}  (python -m AnswerPooling.compare --glob \"<stem>.*.results.jsonl\" "
          f"--exclude-groups {a.out})")
    print(f"-> {a.detail}")


if __name__ == "__main__":
    main()
