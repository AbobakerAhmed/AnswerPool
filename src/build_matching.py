#!/usr/bin/env python
# ============================================================================
# build_matching.py — convert QuALITY MCQ items into matching-format groups.
#
# Construction (see MATCHING_EVAL.md):
#   group N questions from ONE article
#   -> pool their N gold answers          (golds only: no fabricated distractors)
#   -> pad to M > N with golds from OTHER articles   (kills last-item freebie,
#                                                     keeps pool stylistically
#                                                     homogeneous, preserves the
#                                                     choices-only theorem)
#   -> optionally WITHHOLD a fraction p of golds     (variant C: removes the
#                                                     complete-matching
#                                                     assumption; enables the
#                                                     false-match rate)
#
# Emits JSONL, one record per group, with the prompt and the gold assignment.
#
#   python build_matching.py --n 5 --m 8 --withhold 0.2 --out groups_n5.jsonl
#   python build_matching.py --n 5 --m 8 --arm choices_only --out co_n5.jsonl
# ============================================================================
import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict


def norm(s):
    return " ".join(str(s).split()).strip()


def art_key(s):
    return hashlib.sha1(norm(s)[:400].encode("utf-8")).hexdigest()[:16]


def gold_of(row):
    """QuALITY (emozilla/quality): `answer` is a 0-BASED index into `options`.
    Verified: raw values are {0,1,2,3} with 4 options, and a model's picks agree
    with the 0-based reading 82% of the time vs 0% for 1-based
    (`python diagnose.py --check gold`). An off-by-one here is nearly invisible
    in the matching arm -- a wrong option is still topically tied to its own
    question, so matching-by-topic still scores well -- but collapses MCQ
    accuracy to ~0. Do not "fix" this without re-running that diagnostic."""
    opts, a = row["options"], row["answer"]
    if isinstance(a, int) or str(a).strip().isdigit():
        return norm(opts[int(a)])
    s = str(a).strip().upper()
    if s and s[0] in "ABCDEFGH":
        return norm(opts[ord(s[0]) - 65])
    return norm(a)


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NONE_TEXT = "None of these answers is correct."
# datasets with no passage: they group by topic and cannot run the arms that
# manipulate passage availability (closed-book, free-form, the escape arms)
PASSAGE_FREE = {"mmlupro", "gpqa", "ceval"}


def build_prompt(article, questions, candidates, allow_none, with_passage=True):
    """with_passage=False -> the BLIND control: questions + candidates, no passage.
    This is the arm that decides whether matching measures comprehension or merely
    question-answer semantic compatibility. choices_only (no questions either)
    tests the theorem; this tests the construct."""
    q_block = "\n".join(f"{i+1}. {norm(q)}" for i, q in enumerate(questions))
    c_block = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates))
    none_line = (
        f'\nIf a question\'s answer is NOT among the candidates, write "none" for it.'
        if allow_none else ""
    )
    if not article:
        # knowledge datasets (MMLU-Pro): no passage exists, so neither mention
        # one nor apologise for its absence
        head = (f"Match each question to its correct answer from the candidate "
                f"list.\n\nEach candidate may be used at most once.{none_line}\n\n")
    elif with_passage:
        head = (f"Read the passage, then match each question to its correct answer from the "
                f"candidate list.\n\nEach candidate may be used at most once.{none_line}\n\n"
                f"PASSAGE:\n{norm(article)}\n\n")
    else:
        head = (f"Match each question to its correct answer from the candidate list.\n\n"
                f"Each candidate may be used at most once.{none_line}\n"
                f"The source passage is not provided; answer as best you can.\n\n")
    return (
        f"{head}QUESTIONS:\n{q_block}\n\nCANDIDATE ANSWERS:\n{c_block}\n\n"
        f"Respond with one line per question in the form `<question number>: <letter>`"
        f"{' or `<question number>: none`' if allow_none else ''}, and nothing else."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="questions per group")
    ap.add_argument("--m", type=int, default=8, help="candidates in pool (M > N)")
    ap.add_argument("--withhold", type=float, default=0.0,
                    help="fraction of questions whose gold is REMOVED from the pool")
    ap.add_argument("--mirror", default="",
                    help="a built matching .jsonl whose withheld questions this arm "
                         "should copy item-for-item. Required for --arm mcq_none: "
                         "without it the two formats withhold DIFFERENT questions "
                         "and the comparison is unpaired, which throws away most of "
                         "the statistical power and invites a confound.")
    ap.add_argument("--arm", default="matching",
                    choices=["matching", "choices_only", "no_passage", "mcq",
                             "mcq_no_passage", "mcq_choices_only", "mcq_none",
                             "mcq_prose", "freeform", "easy"],
                    help="choices_only drops the passage AND questions (must be chance); "
                         "easy = pad with golds from unrelated articles only (format-tax control)")
    ap.add_argument("--exclude", default="",
                    help="file of question hashes to drop (from filter_blind.py): "
                         "removes items answerable WITHOUT the passage, leaving a "
                         "context-requiring subset")
    ap.add_argument("--distractors", action="store_true",
                    help="pool each question's OWN 3 distractors alongside the golds "
                         "(M = 4N). Golds-only pooling deletes the adversarial defence "
                         "that keeps MCQ hard: candidates become answers to OTHER "
                         "questions, so question-answer compatibility discriminates and "
                         "the passage becomes unnecessary (blind 0.841 vs full 0.879). "
                         "Own-distractors are written to be plausible for THIS question, "
                         "so compatibility carries no signal again.")
    ap.add_argument("--allow-cross-pad", action="store_true",
                    help="fall back to other-article golds if the article lacks spare "
                         "golds (topically obvious -- off by default)")
    ap.add_argument("--race-config", default="all", choices=["all", "high", "middle"],
                    help="race: high has longer passages than middle")
    ap.add_argument("--dataset", default="quality",
                    choices=["quality", "race", "mmlupro", "gpqa", "ceval"],
                    help="mmlupro (TIGER-Lab/MMLU-Pro): 10-option, distractors "
                         "sampled via --max-distractors. gpqa "
                         "(Idavidrein/gpqa, gated): 4-option STEM, groups by "
                         "subdomain. ceval (ceval/ceval-exam val split): "
                         "4-option Chinese, groups by subject. All three are "
                         "passage-free and group by topic")
    ap.add_argument("--group-by", default="src", choices=["src", "category"],
                    help="mmlupro grouping: src = fine source topic (more "
                         "confusable pools), category = 14 coarse subjects")
    ap.add_argument("--max-distractors", type=int, default=3,
                    help="distractors per question admitted to the pool. 3 keeps "
                         "M=4N and matches the QuALITY construction; QuALITY "
                         "items only have 3 so its builds are unchanged")
    ap.add_argument("--splits", default="validation,train")
    ap.add_argument("--max-groups", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="groups.jsonl")
    args = ap.parse_args()
    assert args.m >= args.n, "--m must be >= --n"
    rng = random.Random(args.seed)

    mirror, mirror_recs = {}, []
    if args.mirror:
        for line in open(args.mirror, encoding="utf-8"):
            line = line.strip()
            if line:
                r0 = json.loads(line)
                mirror[r0["group_id"]] = set(r0.get("withheld", []))
                mirror_recs.append(r0)
        print(f"mirroring withheld questions from {args.mirror}: "
              f"{len(mirror)} groups, "
              f"{sum(len(v) for v in mirror.values())} withheld questions")
    if args.arm in ("mcq_none", "mcq_prose", "freeform", "easy") and not mirror:
        raise SystemExit(f"--arm {args.arm} needs --mirror <matching file>; see --help")
    if args.dataset in PASSAGE_FREE:
        if args.arm in ("freeform", "mcq_none", "mcq_prose", "no_passage",
                        "mcq_no_passage"):
            raise SystemExit(f"--arm {args.arm} needs a passage; not applicable "
                             f"to {args.dataset}")
    if args.dataset != "quality":
        if args.arm == "mcq" and not mirror:
            raise SystemExit(f"{args.dataset} mcq must be mirror-built for exact "
                             f"pairing: build the matching file first and pass "
                             f"--mirror <matching file>")

    from datasets import load_dataset
    rows = []
    if args.dataset == "mmlupro":
        # no passage exists: article stays empty, which switches every prompt
        # to its knowledge variant. answer_index is 0-based like QuALITY.
        for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
            rows.append(dict(article="", question=r["question"],
                             options=list(r["options"]),
                             answer=int(r["answer_index"]),
                             _topic=str(r[args.group_by])))
        print(f"loaded {len(rows)} MMLU-Pro questions, grouping by {args.group_by}")
    elif args.dataset == "gpqa":
        # gated on the hub: accept the license and set HF_TOKEN first. Options
        # ship UNORDERED (correct + 3 incorrect fields), so a canonical
        # gold-first order would leak position into the mirror-built MCQ arm.
        # Shuffle per item with the seeded rng at load time.
        for r in load_dataset("Idavidrein/gpqa", "gpqa_main", split="train"):
            opts = [norm(r["Correct Answer"]), norm(r["Incorrect Answer 1"]),
                    norm(r["Incorrect Answer 2"]), norm(r["Incorrect Answer 3"])]
            gold = opts[0]
            rng.shuffle(opts)
            rows.append(dict(article="", question=r["Question"], options=opts,
                             answer=opts.index(gold),
                             _topic=str(r["Subdomain"])))
        print(f"loaded {len(rows)} GPQA questions, grouping by subdomain")
    elif args.dataset == "race":
        # HAS a passage, so it groups by article and supports every instrument
        # arm. Passages are short (a few hundred words) against QuALITY's few
        # thousand, which is the point: it tests the mechanism in a different
        # context regime. Answer ships as a letter.
        for sp in args.splits.split(","):
            for r in load_dataset("ehovy/race", args.race_config, split=sp.strip()):
                gi = LETTERS.find(str(r["answer"]).strip().upper()[:1])
                if not (0 <= gi < len(r["options"])):
                    continue
                rows.append(dict(article=r["article"], question=r["question"],
                                 options=[norm(o) for o in r["options"]],
                                 answer=gi))
        print(f"loaded {len(rows)} RACE questions ({args.race_config})")
    elif args.dataset == "ceval":
        # val split is the largest with public answers. 52 subject configs,
        # each config becomes a topic. Answer is a letter A-D.
        from datasets import get_dataset_config_names
        for subj in get_dataset_config_names("ceval/ceval-exam"):
            for r in load_dataset("ceval/ceval-exam", subj, split="val"):
                opts = [norm(r["A"]), norm(r["B"]), norm(r["C"]), norm(r["D"])]
                gi = LETTERS.find(str(r["answer"]).strip().upper()[:1])
                if not (0 <= gi < 4):
                    continue
                rows.append(dict(article="", question=r["question"],
                                 options=opts, answer=gi, _topic=subj))
        print(f"loaded {len(rows)} C-Eval questions, grouping by subject")
    else:
        for sp in args.splits.split(","):
            rows += list(load_dataset("emozilla/quality", split=sp.strip()))
        print(f"loaded {len(rows)} questions")

    # --- guard against the 0/1-based gold bug ever coming back -------------
    ans_vals, n_opts = set(), set()
    for r in rows:
        a = r["answer"]
        if isinstance(a, int) or str(a).strip().isdigit():
            ans_vals.add(int(a))
        n_opts.add(len(r["options"]))
    if ans_vals:
        lo, hi, k = min(ans_vals), max(ans_vals), max(n_opts)
        print(f"answer index range: [{lo}, {hi}]   options per item: {sorted(n_opts)}")
        assert args.dataset != "quality" or (lo == 0 and hi == k - 1), (
            f"expected 0-based indices spanning [0,{k-1}], got [{lo},{hi}] — "
            f"gold_of() convention may be wrong; run `python diagnose.py --check gold`")

    if args.exclude and args.dataset in PASSAGE_FREE:
        print(f"NOTE: --exclude is a closed-book filter, ignored for "
              f"{args.dataset} which is closed book by nature")
        args.exclude = ""
    if args.exclude and os.path.exists(args.exclude):
        drop = {l.strip() for l in open(args.exclude, encoding="utf-8") if l.strip()}
        before = len(rows)
        rows = [r for r in rows
                if hashlib.sha1(norm(r["question"]).lower().encode()).hexdigest()[:16] not in drop]
        print(f"excluded {before-len(rows)} blind-solvable questions "
              f"({100*(before-len(rows))/before:.1f}%) -> {len(rows)} remain")

    by_article = defaultdict(list)
    for r in rows:
        k = r["_topic"] if args.dataset in PASSAGE_FREE else art_key(r["article"])
        by_article[k].append(r)
    print(f"{len(by_article)} "
          f"{'topics' if args.dataset in PASSAGE_FREE else 'articles'}")

    # global gold bank for cross-article padding
    all_golds = {k: [gold_of(r) for r in v] for k, v in by_article.items()}
    art_text = {k: v[0]["article"] for k, v in by_article.items()}
    art_keys = sorted(art_text)

    if args.arm == "easy":
        # FORMAT-TAX CONTROL, mirror-driven. Same groups and questions as the
        # mirror (use h_match.jsonl), same pool SIZE, but the pool is the
        # group's own golds plus golds of UNRELATED articles. Pads are
        # topically alien, so a model scoring near its MCQ accuracy here
        # certifies that assembling the assignment costs nothing and the drop
        # on real pools is discrimination among plausible candidates.
        out_groups, missing = [], 0
        for r0 in mirror_recs:
            if r0.get("withheld"):
                raise SystemExit("mirror a NO-withhold file (h_match.jsonl) for --arm easy")
            key = r0.get("article_key", "")
            if key not in art_text:
                missing += 1
                continue
            qs = r0.get("questions", [])
            cands0 = r0.get("candidates", [])
            golds = []
            for qi, letter in enumerate(r0.get("answer", [])):
                j = LETTERS.find(letter)
                golds.append(cands0[j] if 0 <= j < len(cands0) else "")
            if any(not g for g in golds) or len(set(golds)) != len(golds):
                missing += 1
                continue
            m_target = len(cands0)              # match the mirror's M exactly
            bank = [g for k2, gs in all_golds.items() if k2 != key for g in gs]
            rng.shuffle(bank)
            pad, seen = [], set(golds)
            for g in bank:
                if len(pad) >= m_target - len(golds):
                    break
                if g and g not in seen:
                    pad.append(g); seen.add(g)
            cand = golds + pad
            rng.shuffle(cand)
            pos = {c: i for i, c in enumerate(cand)}
            answer = [LETTERS[pos[g]] for g in golds]
            out_groups.append(dict(
                group_id=r0["group_id"], article_key=key, n=len(qs), m=len(cand),
                arm="easy", withheld=[], n_cross_pad=len(pad), none_letter="",
                questions=qs, candidates=cand, answer=answer,
                prompt=build_prompt(art_text[key], qs, cand,
                                    allow_none=False, with_passage=True)))
        with open(args.out, "w", encoding="utf-8") as f:
            for g in out_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(out_groups)} easy groups -> {args.out} "
              f"({missing} mirror groups skipped)")
        print("expectation: accuracy near the same models' MCQ accuracy. A gap "
              "here is format tax and bounds how much of the real-pool drop "
              "the format itself explains.")
        return

    if args.arm in ("mcq", "mcq_no_passage") and mirror:
        # Mirror-driven plain MCQ: the SAME questions as the matching file,
        # each with its own original options. Independently built mcq and
        # matching files share group_ids but not necessarily questions (the
        # per-group rng consumption differs between arms, which reshuffles
        # later articles), so exact question-level pairing requires building
        # from the mirror. Original option order is preserved, no shuffle.
        by_q = {}
        for r in rows:
            by_q.setdefault(norm(r["question"]), r)
        out_groups, missing = [], 0
        for r0 in mirror_recs:
            for qi, q in enumerate(r0.get("questions", [])):
                row = by_q.get(q)
                if row is None:
                    missing += 1
                    continue
                opts = [norm(o) for o in row["options"]]
                gold = gold_of(row)
                gi = opts.index(gold)
                blk = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
                art = row.get("article", "")
                if args.arm == "mcq_no_passage":
                    p = (f"Answer the question. The source passage is not "
                         f"provided; answer as best you can.\n\n"
                         f"QUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
                         f"Respond with only the letter of the correct option.")
                elif art:
                    p = (f"Read the passage and answer the question.\n\n"
                         f"PASSAGE:\n{norm(art)}\n\n"
                         f"QUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
                         f"Respond with only the letter of the correct option.")
                else:
                    p = (f"Answer the question.\n\n"
                         f"QUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
                         f"Respond with only the letter of the correct option.")
                out_groups.append(dict(
                    group_id=r0["group_id"], item_id=f"{r0['group_id']}_q{qi}",
                    q_index=qi, n=1, m=len(opts), arm=args.arm, withheld=[],
                    none_letter="", n_cross_pad=0, questions=[q],
                    candidates=opts, answer=[LETTERS[gi]], prompt=p))
        with open(args.out, "w", encoding="utf-8") as f:
            for g in out_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(out_groups)} mirror-paired mcq items -> {args.out} "
              f"({missing} questions not found)")
        return

    if args.arm in ("mcq_none", "mcq_prose"):
        # Also mirror-driven. Regenerating groups from the dataset produced a
        # file that LOOKED right (1535 items, 40% withheld) but shared only 85
        # unanswerable items with the matching arm, because rng.shuffle(items)
        # chunked the articles differently -- same group_id, different questions
        # inside it. Join on question TEXT so the two arms are the same items by
        # construction. (Question text is needed because the mirror stores the
        # group's pooled candidates, not each question's own four options.)
        #
        # mcq_none  = printed escape: "none of these" is a lettered option.
        # mcq_prose = DECOMPOSITION arm: no printed option, only the one-line
        #   prose permission the matching arm uses. mcq_none differs from
        #   matching in two ways at once (independent items vs competitive
        #   pool, printed option vs prose permission). This arm isolates the
        #   printed-option effect: identical items, identical 3-option count,
        #   only the escape's FORM changes.
        by_q = {}
        for r in rows:
            by_q.setdefault(norm(r["question"]), r)
        out_groups, missing = [], 0
        for r0 in mirror_recs:
            wh = set(r0.get("withheld", []))
            for qi, q in enumerate(r0.get("questions", [])):
                row = by_q.get(q)
                if row is None:
                    missing += 1
                    continue
                gold = gold_of(row)
                dis = [o for o in (norm(x) for x in row["options"]) if o != gold]
                rng.shuffle(dis)
                bad = qi in wh
                cand = (dis[:3] if bad else [gold] + dis[:2])
                if args.arm == "mcq_none":
                    cand = cand + [NONE_TEXT]
                rng.shuffle(cand)
                gi = -1 if bad else cand.index(gold)
                blk = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(cand))
                if args.arm == "mcq_none":
                    none_letter = LETTERS[cand.index(NONE_TEXT)]
                    p = (f"Read the passage and answer the question.\n\n"
                         f"PASSAGE:\n{norm(row['article'])}\n\n"
                         f"QUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
                         f"One of the options states that none of the listed "
                         f"answers is correct. Choose it if, and only if, that "
                         f"is the case.\n"
                         f"Respond with only the letter of the correct option.")
                else:
                    none_letter = ""
                    # mirrors the matching arm's permission line as closely as
                    # a single-question prompt allows
                    p = (f"Read the passage and answer the question.\n\n"
                         f"PASSAGE:\n{norm(row['article'])}\n\n"
                         f"QUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
                         f'If the correct answer is NOT among the options, '
                         f'write "none".\n'
                         f"Respond with only the letter of the correct option, "
                         f"or the word none.")
                out_groups.append(dict(
                    group_id=r0["group_id"], item_id=f"{r0['group_id']}_q{qi}",
                    q_index=qi, n=1, m=len(cand), arm=args.arm,
                    withheld=([0] if bad else []),
                    none_letter=none_letter, n_cross_pad=0,
                    questions=[q], candidates=cand,
                    answer=(["none"] if bad else [LETTERS[gi]]),
                    prompt=p))
        with open(args.out, "w", encoding="utf-8") as f:
            for g in out_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        n_bad = sum(1 for g in out_groups if g["withheld"])
        print(f"\nwrote {len(out_groups)} {args.arm} items -> {args.out}")
        print(f"  from {len(mirror_recs)} mirrored groups: {n_bad} unanswerable, "
              f"{len(out_groups)-n_bad} answerable, {missing} questions not found")
        return

    if args.arm == "freeform":
        # Built ENTIRELY from the mirror file, never regenerated from the
        # dataset. Regenerating cannot be kept aligned: rng.shuffle(items) runs
        # per article, so any difference in how much randomness earlier articles
        # consume reshuffles the chunking and silently produces different groups
        # under the same group_id. Reading the mirror makes alignment exact by
        # construction rather than by luck.
        out_groups = []
        for r0 in mirror_recs:
            key = r0.get("article_key", "")
            wh = set(r0.get("withheld", []))
            qs = r0.get("questions", [])
            cands = r0.get("candidates", [])
            # deterministic mismatch: the next article in sorted order
            other = (art_keys[(art_keys.index(key) + 1) % len(art_keys)]
                     if key in art_keys and len(art_keys) > 1 else key)
            for qi, q in enumerate(qs):
                bad = qi in wh
                src = other if bad else key
                if src not in art_text:
                    continue
                a0 = r0.get("answer", [])
                gold = ""
                if not bad and qi < len(a0) and a0[qi] != "none":
                    j = LETTERS.find(a0[qi])
                    gold = cands[j] if 0 <= j < len(cands) else ""
                out_groups.append(dict(
                    group_id=r0["group_id"], item_id=f"{r0['group_id']}_q{qi}",
                    q_index=qi, n=1, m=0, arm="freeform",
                    withheld=([0] if bad else []), passage_key=src,
                    n_cross_pad=0, none_letter="", questions=[q], candidates=[],
                    answer=(["none"] if bad else [gold]),
                    prompt=(f"Answer the question using only the passage below.\n\n"
                            f"PASSAGE:\n{norm(art_text[src])}\n\n"
                            f"QUESTION: {q}\n\n"
                            f"Answer in one or two sentences.")))
        with open(args.out, "w", encoding="utf-8") as f:
            for g in out_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        n_bad = sum(1 for g in out_groups if g["withheld"])
        print(f"\nwrote {len(out_groups)} freeform items -> {args.out}")
        print(f"  from {len(mirror_recs)} mirrored groups: {n_bad} unanswerable "
              f"(mismatched passage), {len(out_groups)-n_bad} answerable")
        print("  NOTE: the prompt gives NO abstention instruction, by design — "
              "telling the model it may decline would rebuild mcq_none's offered\n"
              "  escape hatch and the criterion would stop being independent.")
        return

    groups, skipped_dup, skipped_alpha = [], 0, 0
    for key, items in by_article.items():
        rng.shuffle(items)
        for start in range(0, len(items) - args.n + 1, args.n):
            chunk = items[start:start + args.n]
            golds = [gold_of(r) for r in chunk]
            if len(set(golds)) != len(golds):        # answer-uniqueness screen
                skipped_dup += 1
                continue

            # withhold: these questions have NO answer in the pool
            k_withheld = int(round(args.withhold * args.n))
            withheld_idx = set(rng.sample(range(args.n), k_withheld)) if k_withheld else set()
            present = [g for i, g in enumerate(golds) if i not in withheld_idx]

            # MCQ arms must be checked FIRST: they use each question's own four
            # options, so --distractors is meaningless there. Letting the
            # distractors branch run first silently emitted matching records
            # under an mcq arm name.
            if (args.distractors and not args.arm.startswith("mcq")
                    and args.arm != "freeform"):
                # Pool = every option of every question in the group: N golds +
                # 3N adversarial distractors. Each question's own distractors are
                # present, so "which candidate is shaped like an answer to this
                # question" no longer discriminates.
                cand = list(present)
                seen_d = set(cand)
                for r in chunk:
                    g = gold_of(r)
                    dis = [norm(o) for o in r["options"] if norm(o) != g]
                    # 10-option datasets would pool 10N candidates and blow past
                    # the 26-letter alphabet: admit a sample per question. With
                    # 3 distractors (QuALITY) the sample is the whole list and
                    # no rng is consumed, so existing builds reproduce exactly.
                    if len(dis) > args.max_distractors:
                        dis = rng.sample(dis, args.max_distractors)
                    for o in dis:
                        if o not in seen_d:
                            cand.append(o); seen_d.add(o)
                # the answer alphabet ends at Z: at N=7+ the 4N pool can
                # exceed 26 and LETTERS[i] would crash mid-build. Skip and
                # count instead -- and cap the N sweep at 6.
                if len(cand) > 26:
                    skipped_alpha += 1
                    continue
                rng.shuffle(cand)
                pos = {c: i for i, c in enumerate(cand)}
                answer = [("none" if i in withheld_idx else LETTERS[pos[golds[i]]])
                          for i in range(args.n)]
                questions = [norm(r["question"]) for r in chunk]
                article = chunk[0]["article"]
                prompt = build_prompt(article, questions, cand,
                                      allow_none=bool(withheld_idx),
                                      with_passage=(args.arm != "no_passage"))
                if args.arm == "choices_only":
                    prompt = (
                        "This is a control condition. You are shown ONLY a list of candidate "
                        "answers; the passage and the questions have been deliberately withheld.\n\n"
                        + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cand))
                        + f"\n\nThere are {args.n} hidden questions. Assign a distinct candidate "
                          f"letter to each one.\n\nYou are expected to guess — there is no way to "
                          f"know the correct assignment from this information, and that is the "
                          f"point of the control. Do not refuse, apologise, or explain.\n\n"
                          f"Output exactly {args.n} lines in the form `<number>: <letter>`, "
                          f"and nothing else.")
                groups.append(dict(
                    group_id=f"{key}_{start}", article_key=key, n=args.n, m=len(cand),
                    arm=args.arm, withheld=sorted(withheld_idx), n_cross_pad=0,
                    questions=questions, candidates=cand, answer=answer, prompt=prompt))
                if args.max_groups and len(groups) >= args.max_groups:
                    break
                continue

            # ---- padding -------------------------------------------------
            # Pad with golds from the SAME article (other questions, not in this
            # group). Cross-article padding is topically obvious -- a model can
            # filter it in one pass, collapsing M back to N and restoring the
            # bijection + set-level elimination, and it partially leaks into the
            # choices-only arm. Same-article golds share topic, entities and
            # style, so the pool is genuinely homogeneous.
            # MCQ arms use each question's OWN options and build no pool, so they
            # must skip this section entirely. They used to fall through it, which
            # was invisible on QuALITY (13 to 20 questions per article always
            # yield spare golds) and fatal on any corpus with few questions per
            # document: padding needs m-N spares, RACE articles have 3 to 6
            # questions total, so the insufficient-pad `continue` below fired for
            # every group and the mcq branch was never reached.
            n_cross, candidates, answer = 0, [], []
            if not args.arm.startswith("mcq"):
                need = args.m - len(present)
                seen = set(present)
                in_group_ids = {id(r) for r in chunk}
                same_art = [gold_of(r) for r in items if id(r) not in in_group_ids]
                rng.shuffle(same_art)
                pad = []
                for c in same_art:
                    if len(pad) >= need:
                        break
                    if c not in seen:
                        pad.append(c); seen.add(c)
                if len(pad) < need and args.allow_cross_pad:   # fallback only
                    pool_src = [g for k2, gs in all_golds.items() if k2 != key for g in gs]
                    while len(pad) < need and pool_src:
                        c = rng.choice(pool_src)
                        if c not in seen:
                            pad.append(c); seen.add(c)
                if len(pad) < need:        # not enough same-article spares
                    continue
                n_cross = max(0, need - len([c for c in pad if c in set(same_art)]))
                candidates = present + pad
                rng.shuffle(candidates)

                pos = {c: i for i, c in enumerate(candidates)}
                answer = [("none" if i in withheld_idx else LETTERS[pos[golds[i]]])
                          for i in range(args.n)]

            questions = [norm(r["question"]) for r in chunk]
            article = chunk[0]["article"]

            if args.arm == "freeform":
                # THE EXTERNAL CRITERION. Matching and mcq_none disagree about
                # models; neither can arbitrate its own verdict, so we need a
                # measure of the behaviour we actually care about -- does the
                # model invent an answer the evidence does not support.
                #
                # Unanswerability is made the way it fails in deployment: the
                # question is paired with a DIFFERENT article, exactly a RAG
                # retrieval miss. Automatic, no annotation.
                #
                # NO ABSTENTION INSTRUCTION. The prompt must not hint that
                # declining is an option -- telling the model it may say "not in
                # the passage" rebuilds mcq_none's offered escape hatch in prose
                # and the criterion stops being independent of the thing it is
                # supposed to arbitrate.
                wh_mirror = mirror.get(f"{key}_{start}", set())
                others = [k for k in art_keys if k != key]
                for qi, r in enumerate(chunk):
                    unanswerable = qi in wh_mirror
                    src = rng.choice(others) if (unanswerable and others) else key
                    p = (f"Answer the question using only the passage below.\n\n"
                         f"PASSAGE:\n{norm(art_text[src])}\n\n"
                         f"QUESTION: {questions[qi]}\n\n"
                         f"Answer in one or two sentences.")
                    groups.append(dict(
                        group_id=f"{key}_{start}", item_id=f"{key}_{start}_q{qi}",
                        q_index=qi, n=1, m=0, arm="freeform",
                        withheld=([0] if unanswerable else []),
                        passage_key=src, n_cross_pad=0, none_letter="",
                        questions=[questions[qi]], candidates=[],
                        answer=(["none"] if unanswerable else [gold_of(r)]),
                        prompt=p))
                if args.max_groups and len(groups) >= args.max_groups:
                    break
                continue

            if args.arm in ("mcq", "mcq_no_passage", "mcq_choices_only", "mcq_none"):
                # PAIRED BASELINE: the same N questions as standard MCQ, with
                # their ORIGINAL options. One record per question so the
                # comparison is on identical items, aggregatable by group_id.
                wh_mirror = mirror.get(f"{key}_{start}", set())
                for qi, r in enumerate(chunk):
                    opts = [norm(o) for o in r["options"]]
                    gold = gold_of(r)
                    gi = opts.index(gold)
                    none_letter = ""
                    if args.arm == "mcq_none":
                        # DELETE-GOLD MCQ — the per-question abstention baseline.
                        # Two properties are load-bearing and easy to get wrong:
                        #  * "none" appears in EVERY item. Offer it only where the
                        #    gold was deleted and its mere presence gives the answer
                        #    away, measuring nothing.
                        #  * the option count is CONSTANT (4). Leaving the gold in
                        #    and appending "none" makes answerable items 5-way and
                        #    unanswerable ones 4-way, which leaks just as badly, so
                        #    we drop one distractor to make room instead.
                        # Withheld questions mirror the matching arm item-for-item,
                        # so the two formats are paired on the same questions.
                        absent = qi in wh_mirror
                        dis = [o for o in opts if o != gold]
                        rng.shuffle(dis)
                        opts = (dis[:3] if absent else [gold] + dis[:2]) + [NONE_TEXT]
                        rng.shuffle(opts)
                        none_letter = LETTERS[opts.index(NONE_TEXT)]
                        gi = -1 if absent else opts.index(gold)
                    blk = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
                    if args.arm == "mcq_none":
                        p = (f"Read the passage and answer the question.\n\n"
                             f"PASSAGE:\n{norm(article)}\n\n"
                             f"QUESTION: {questions[qi]}\n\nOPTIONS:\n{blk}\n\n"
                             f"One of the options states that none of the listed "
                             f"answers is correct. Choose it if, and only if, that "
                             f"is the case.\n"
                             f"Respond with only the letter of the correct option.")
                    elif args.arm == "mcq":
                        p = (f"Read the passage and answer the question.\n\n"
                             f"PASSAGE:\n{norm(article)}\n\n"
                             f"QUESTION: {questions[qi]}\n\nOPTIONS:\n{blk}\n\n"
                             f"Respond with only the letter of the correct option.")
                    elif args.arm == "mcq_no_passage":       # blind MCQ control
                        p = (f"Answer the question. The source passage is not "
                             f"provided; answer as best you can.\n\n"
                             f"QUESTION: {questions[qi]}\n\nOPTIONS:\n{blk}\n\n"
                             f"Respond with only the letter of the correct option.")
                    else:
                        # ACL'24 replication: options ONLY, no question, no passage.
                        # This is the artifact our choices-only arm must be compared
                        # against -- without it, "matching leaks +0.044" has no scale.
                        p = (f"This is a control condition. You are shown ONLY the "
                             f"answer options; the passage and the question have been "
                             f"deliberately withheld.\n\nOPTIONS:\n{blk}\n\n"
                             f"Exactly one option is the correct answer to the hidden "
                             f"question. You are expected to guess. Do not refuse or "
                             f"explain.\n\nRespond with only the letter.")
                    groups.append(dict(
                        group_id=f"{key}_{start}", item_id=f"{key}_{start}_q{qi}",
                        q_index=qi, n=1, m=len(opts), arm=args.arm,
                        withheld=([0] if gi < 0 else []), none_letter=none_letter,
                        n_cross_pad=0, questions=[questions[qi]], candidates=opts,
                        answer=[("none" if gi < 0 else LETTERS[gi])], prompt=p))
                if args.max_groups and len(groups) >= args.max_groups:
                    break
                continue

            if args.arm == "choices_only":
                # Partial-input control. Must be stated explicitly, or the model
                # treats a question-less prompt as malformed and refuses --
                # refusals then masquerade as failures (17% unparsed before this).
                prompt = (
                    "This is a control condition. You are shown ONLY a list of candidate "
                    "answers; the passage and the questions have been deliberately withheld.\n\n"
                    + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates))
                    + f"\n\nThere are {args.n} hidden questions. Assign a distinct candidate "
                      f"letter to each one.\n\nYou are expected to guess — there is no way to "
                      f"know the correct assignment from this information, and that is the "
                      f"point of the control. Do not refuse, apologise, or explain.\n\n"
                      f"Output exactly {args.n} lines in the form `<number>: <letter>`, "
                      f"and nothing else.")
            else:
                prompt = build_prompt(article, questions, candidates,
                                      allow_none=bool(withheld_idx),
                                      with_passage=(args.arm != "no_passage"))

            groups.append(dict(
                group_id=f"{key}_{start}", article_key=key, n=args.n, m=args.m,
                arm=args.arm, withheld=sorted(withheld_idx), n_cross_pad=n_cross,
                questions=questions, candidates=candidates,
                answer=answer, prompt=prompt))
            if args.max_groups and len(groups) >= args.max_groups:
                break
        if args.max_groups and len(groups) >= args.max_groups:
            break

    with open(args.out, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # report the ACTUAL n/m from the emitted records, not the CLI args:
    # --distractors overrides m, and the mcq arms emit n=1 (one question per
    # record) with m=4, which made prod(range(m-n+1, m+1)) collapse to zero.
    n = groups[0]["n"] if groups else args.n
    m = groups[0]["m"] if groups else args.m
    inj = math.prod(range(max(1, m - n + 1), m + 1)) if m >= n else m
    cross = sum(g["n_cross_pad"] for g in groups)
    print(f"\nwrote {len(groups)} groups -> {args.out}"
          f"   (skipped {skipped_dup} for duplicate golds"
          f"{f', {skipped_alpha} for pool exceeding 26 letters' if skipped_alpha else ''})")
    print(f"padding: {cross} cross-article candidates total "
          f"({'GOOD - all same-article' if cross == 0 else 'WARNING - topically filterable'})")
    print(f"floors at N={n}, M={m}:  per-pair 1/{m} = {1/m:.4f}   "
          f"exact-assignment 1/{inj:,} = {1/inj:.2e}")
    if groups:
        print("\n--- sample prompt (truncated) ---")
        p = groups[0]["prompt"]
        print(p[:600] + ("\n... [passage truncated] ...\n" + p[-500:] if len(p) > 1100 else ""))
        print(f"--- gold: {groups[0]['answer']}")


if __name__ == "__main__":
    main()
