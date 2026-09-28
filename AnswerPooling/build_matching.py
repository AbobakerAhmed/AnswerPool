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
#   python -m AnswerPooling.build_matching --n 5 --m 8 --withhold 0.2 --out groups_n5.jsonl
#   python -m AnswerPooling.build_matching --n 5 --m 8 --arm choices_only --out co_n5.jsonl
# ============================================================================
import argparse
import hashlib
import json
import re
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
    with the 0-based reading 82% of the time vs 0% for 1-based. An off-by-one
    here is nearly invisible in the matching arm -- a wrong option is still
    topically tied to its own question, so matching-by-topic still scores
    well -- but collapses MCQ accuracy to ~0."""
    opts, a = row["options"], row["answer"]
    if isinstance(a, int) or str(a).strip().isdigit():
        return norm(opts[int(a)])
    s = str(a).strip().upper()
    if s and s[0] in "ABCDEFGH":
        return norm(opts[ord(s[0]) - 65])
    return norm(a)


from .run_matching import LETTERS, label, label_index

NONE_TEXT = "None of these answers is correct."
# datasets with no passage: they group by topic and cannot run the arms that
# manipulate passage availability (closed-book, free-form, the escape arms)
PASSAGE_FREE = {"mmlupro", "gpqa", "ceval", "mmmupro", "medxpertqa", "videomme", "lvbench"}
# of those, the ones with NO context at all (the others carry images or video
# frames, so the no-context screen applies to them through --drop-images)
CLOSED_BOOK = {"mmlupro", "gpqa", "ceval"}
VIDEO = {"videomme", "lvbench"}

# MMMU-Pro questions cite their images inline as "<image 1>", numbered relative
# to each question. Pooling N questions puts every image in one prompt, so the
# markers have to be renumbered into a single global sequence: otherwise
# question 2's "<image 1>" names question 1's picture and the item silently
# becomes unanswerable.
IMGREF = re.compile(r"<\s*image\s*(\d+)\s*>", re.I)


def qkey(q):
    """Join key for mirror-built arms. A pooled question's <image k> markers are
    renumbered globally, so its text no longer equals the source row's and a
    raw-text join would miss every multimodal item. Canonicalise the markers
    away. Identical to norm() on every text-only dataset."""
    return IMGREF.sub("<image>", norm(q))


def renumber_images(questions, per_q_images):
    """Rewrite local <image k> markers to one global sequence and flatten the
    image lists in the same order, so marker j always names the j-th image
    handed to the model. Out-of-range markers are left alone rather than
    remapped to a wrong picture."""
    out_q, out_imgs = [], []
    for q, imgs in zip(questions, per_q_images):
        base = len(out_imgs)
        out_q.append(IMGREF.sub(
            lambda mo: (f"<image {base + int(mo.group(1))}>"
                        if 1 <= int(mo.group(1)) <= len(imgs) else mo.group(0)),
            q))
        out_imgs.extend(imgs)
    return out_q, out_imgs


# Two prompt ablations, set from the CLI. Both change the prompt text only, so
# each produces a new phash and never collides with the default builds.
#   EXCLUSIVE=False drops the at-most-once rule: same pool, options may be
#     reused. Isolates the exclusivity constraint from the pool size.
#   DECLINE_WORDING="alt" rephrases the permission to answer none, to show the
#     false-answer rate is not an artifact of one sentence.
EXCLUSIVE = True
DECLINE_WORDING = "default"
DECLINE_LINES = {
    "default": '\nIf a question\'s answer is NOT among the candidates, write "none" for it.',
    "alt": '\nSome questions may have no correct answer in the list. Answer "none" for those.',
}


def video_note(n_frames):
    """One line that tells the model what the attached images are."""
    return (f"The video is given as {n_frames} frames in order, attached before "
            f"this text.")


def build_prompt(article, questions, candidates, allow_none, with_passage=True,
                 context_note=""):
    """with_passage=False -> the BLIND control: questions + candidates, no passage.
    This is the arm that decides whether matching measures comprehension or merely
    question-answer semantic compatibility. choices_only (no questions either)
    tests the theorem; this tests the construct.
    context_note: for video groups, the sentence that explains the attached
    frames; it replaces the passage block."""
    q_block = "\n".join(f"{i+1}. {norm(q)}" for i, q in enumerate(questions))
    c_block = "\n".join(f"{label(i)}. {c}" for i, c in enumerate(candidates))
    none_line = DECLINE_LINES[DECLINE_WORDING] if allow_none else ""
    rule = ("Each candidate may be used at most once." if EXCLUSIVE
            else "A candidate may be the answer to more than one question.")
    if context_note:
        head = (f"{context_note} Match each question to its correct answer from "
                f"the candidate list.\n\n{rule}{none_line}\n\n")
    elif not article:
        # knowledge datasets (MMLU-Pro): no passage exists, so neither mention
        # one nor apologise for its absence
        head = (f"Match each question to its correct answer from the candidate "
                f"list.\n\n{rule}{none_line}\n\n")
    elif with_passage:
        # a cross-domain group of a passage benchmark carries one passage per
        # question, labelled [Passage k]; the questions cite their passage
        many = article.lstrip().startswith("[Passage 1]")
        head = (f"Read the passage{'s' if many else ''}, then match each question "
                f"to its correct answer from the candidate list.\n\n{rule}{none_line}\n\n"
                f"PASSAGE{'S' if many else ''}:\n{norm(article) if not many else article.strip()}\n\n")
    else:
        head = (f"Match each question to its correct answer from the candidate list.\n\n"
                f"{rule}{none_line}\n"
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
    ap.add_argument("--only", default="",
                    help="file of question hashes to KEEP (the *.passed.txt that "
                         "filter_blind.py writes): restricts the build to questions "
                         "the no-context filter screened and passed, so unscreened "
                         "questions cannot enter a group. Use it whenever the "
                         "screen covered a subset of the benchmark")
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
    ap.add_argument("--drop-images", action="store_true",
                    help="strip the images but keep the question text and its "
                         "<image k> citations. This is the closed-book screen "
                         "for a multimodal benchmark: on MMMU-Pro it is the "
                         "same test its authors used to filter items answerable "
                         "by text-only models, so running it here is a positive "
                         "control, it should flag far less than on QuALITY")
    ap.add_argument("--mmmu-options", type=int, default=10, choices=[4, 10],
                    help="mmmupro: which standard config to load. 10 is the "
                         "augmented-option set, whose 1/10 floor is what the "
                         "transform is being compared against. The `vision` "
                         "config is deliberately not offered: it renders the "
                         "question AND its options into a screenshot, so there "
                         "are no option strings to pool and re-rendering a "
                         "pooled list would be authoring")
    ap.add_argument("--cross-domain", action="store_true",
                    help="cross-domain groups: regroup the questions of the "
                         "--mirror build so that no two questions in a group "
                         "share a context (topic, article, subject). Each "
                         "question keeps its own full option set, so the pool "
                         "size, prompt shape and grammar match the mirror, and "
                         "its neighbours come from unrelated contexts. With "
                         "--withhold or a removed-mode mirror, the removed "
                         "questions are the mirror's")
    ap.add_argument("--keep-slot-order", action="store_true",
                    help="choices_only: do NOT shuffle the hidden-question "
                         "slots. By default the slots are assigned to the "
                         "group's questions in a random order, so slot position "
                         "carries no information")
    ap.add_argument("--frames-index", default="",
                    help="videomme / lvbench: the json written by "
                         "prepare_video.py, video key -> list of frame paths")
    ap.add_argument("--dataset", default="quality",
                    choices=["quality", "race", "mmlupro", "gpqa", "ceval",
                             "mmmupro", "medxpertqa", "videomme", "lvbench"],
                    help="mmlupro (TIGER-Lab/MMLU-Pro): 10-option, distractors "
                         "sampled via --max-distractors. gpqa "
                         "(Idavidrein/gpqa, gated): 4-option STEM, groups by "
                         "subdomain. ceval (ceval/ceval-exam val split): "
                         "4-option Chinese, groups by subject. mmmupro "
                         "(MMMU/MMMU_Pro, multimodal): 10-option, groups by "
                         "subject, carries images. All four are passage-free "
                         "and group by topic")
    ap.add_argument("--group-by", default="src", choices=["src", "category"],
                    help="mmlupro grouping: src = fine source topic (more "
                         "confusable pools), category = 14 coarse subjects")
    ap.add_argument("--max-distractors", type=int, default=3,
                    help="distractors per question admitted to the pool. 3 keeps "
                         "M=4N and matches the QuALITY construction; QuALITY "
                         "items only have 3 so its builds are unchanged")
    ap.add_argument("--splits", default="validation,train")
    ap.add_argument("--revision", default="",
                    help="Hub revision (commit sha) of the source dataset, as "
                         "recorded in a released manifest, to regenerate that "
                         "build exactly. Empty = the latest revision")
    ap.add_argument("--max-groups", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-reuse", action="store_true",
                    help="ablation: drop the at-most-once rule from the pooled "
                         "prompt so options may be reused. Same pool, same "
                         "scoring. Isolates exclusivity from pool size")
    ap.add_argument("--decline-wording", default="default",
                    choices=sorted(DECLINE_LINES),
                    help="ablation: alternative phrasing of the permission to "
                         "answer none on the removal setting")
    ap.add_argument("--out", default="groups.jsonl")
    args = ap.parse_args()
    global EXCLUSIVE, DECLINE_WORDING
    EXCLUSIVE = not args.allow_reuse
    DECLINE_WORDING = args.decline_wording
    if args.allow_reuse or args.decline_wording != "default":
        print(f"prompt ablation: exclusive={EXCLUSIVE} "
              f"decline_wording={DECLINE_WORDING}")
    assert args.m >= args.n, "--m must be >= --n"
    rng = random.Random(args.seed)
    # a separate stream for the options-only slot order, so that shuffling the
    # slots does not change how later articles are chunked (the shared rng is
    # what keeps h_co aligned with h_match)
    rng_slot = random.Random(args.seed + 7919)

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

    if args.arm == "no_passage" and mirror_recs:
        # The pooled no-context arm, mirror-driven. Regenerating it from the
        # dataset reproduces the mirror's groups only if every flag that feeds
        # the chunking (the --exclude list above all) is identical; a different
        # exclude list gives the same group_ids with different questions inside.
        # Built from the mirror, each group keeps its questions, its pool in the
        # same order, and its answer letters. Only the passage goes. No dataset
        # is loaded.
        out_groups = []
        for r0 in mirror_recs:
            if r0.get("withheld"):
                raise SystemExit("mirror a full-pool build (h_match.jsonl), not a "
                                 "removed-mode one")
            if r0.get("images"):
                raise SystemExit("the mirror carries images; use --drop-images on "
                                 "the image and video benchmarks instead")
            g = dict(r0)
            # any non-empty article selects the passage-free wording of
            # build_prompt ("The source passage is not provided ...")
            g.update(arm="no_passage",
                     prompt=build_prompt("(omitted)", r0["questions"], r0["candidates"],
                                         allow_none=False, with_passage=False))
            out_groups.append(g)
        with open(args.out, "w", encoding="utf-8") as f:
            for g in out_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(out_groups)} mirror-paired no-passage groups -> {args.out} "
              f"(same questions, pool and letters as {args.mirror}, passage removed)")
        return

    from datasets import load_dataset
    # pin the source dataset when a revision is given; no extra argument otherwise
    rev = {"revision": args.revision} if args.revision else {}
    rows = []
    if args.dataset == "mmlupro":
        # no passage exists: article stays empty, which switches every prompt
        # to its knowledge variant. answer_index is 0-based like QuALITY.
        for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test", **rev):
            rows.append(dict(article="", question=r["question"],
                             options=list(r["options"]),
                             answer=int(r["answer_index"]),
                             _topic=str(r[args.group_by])))
        print(f"loaded {len(rows)} MMLU-Pro questions, grouping by {args.group_by}")
    elif args.dataset == "mmmupro":
        # MMMU-Pro is the multimodal MMLU-Pro, and its construction is two of
        # our screens done by hand: it filters items answerable by text-only
        # models, then augments 4 options to 10. That makes it the sharpest
        # available foil, and a positive control for the closed-book screen,
        # which should find little here and does find ~50% on QuALITY.
        #
        # Two structural hazards, both handled:
        #   (1) an option can itself be an image citation ("<image 3>"), which
        #       is meaningless once pooled beside another question's options.
        #       Those items are dropped, and the count is reported because it
        #       biases the surviving set toward text-answer questions.
        #   (2) images are PIL objects and cannot live in JSONL. They are
        #       written next to the output file and referenced by path.
        import ast
        cfg = f"standard ({args.mmmu_options} options)"
        # one shared directory per dataset, next to the outputs: every arm and
        # every rebuild reuses the same files instead of extracting a copy each
        img_dir = os.path.join(os.path.dirname(args.out) or ".",
                               f"mmmupro{args.mmmu_options}_images")
        os.makedirs(img_dir, exist_ok=True)
        n_imgopt = n_noimg = n_badgold = 0
        for r in load_dataset("MMMU/MMMU_Pro", cfg, split="test", **rev):
            opts = r["options"]
            if isinstance(opts, str):
                opts = ast.literal_eval(opts)
            opts = [norm(o) for o in opts]
            if any(IMGREF.search(o) for o in opts):
                n_imgopt += 1
                continue
            gi = LETTERS.find(str(r["answer"]).strip().upper()[:1])
            if not (0 <= gi < len(opts)):
                n_badgold += 1
                continue
            paths = []
            for k in range(1, 8):
                im = r.get(f"image_{k}")
                if im is None:
                    continue
                p = os.path.join(img_dir, f"{r['id']}_{k}.png")
                if not os.path.exists(p):
                    im.convert("RGB").save(p)
                paths.append(p)
            if not paths:
                n_noimg += 1
                continue
            rows.append(dict(article="", question=norm(r["question"]),
                             options=opts, answer=gi,
                             _topic=str(r["subject"]), _images=paths))
        print(f"loaded {len(rows)} MMMU-Pro questions ({cfg}), "
              f"grouping by subject")
        print(f"  dropped {n_imgopt} with image-valued options "
              f"(not poolable), {n_noimg} with no image, "
              f"{n_badgold} with an unreadable gold")
        print(f"  images -> {img_dir}/")
    elif args.dataset == "medxpertqa":
        # MedXpertQA MM (ICML 2025): 2,005 test questions, five options, expert
        # written, images in images.zip at the repo root. The question text
        # embeds the options after "Answer Choices:", which is cut off, and the
        # images are not cited in the text, so a citation line is prepended so
        # that pooling renumbers them exactly as it does for MMMU-Pro. Groups
        # are formed by body system (11), the closest analogue of a subject.
        import zipfile
        from huggingface_hub import hf_hub_download
        img_dir = os.path.join(os.path.dirname(args.out) or ".", "medxpertqa_images")
        if not os.path.isdir(img_dir) or not os.listdir(img_dir):
            zp = hf_hub_download("TsinghuaC3I/MedXpertQA", "images.zip", repo_type="dataset", **rev)
            os.makedirs(img_dir, exist_ok=True)
            with zipfile.ZipFile(zp) as z:
                z.extractall(img_dir)
            print(f"  extracted images.zip -> {img_dir}/")
        idx = {}
        for root, _, files in os.walk(img_dir):
            for fn in files:
                idx[fn] = os.path.join(root, fn)
        jp = hf_hub_download("TsinghuaC3I/MedXpertQA", "MM/test.jsonl", repo_type="dataset", **rev)
        # options that only name a picture ("Figure A", "Image 2", "Panel C")
        # are image citations in disguise: meaningless once pooled beside
        # another question's answers, exactly like MMMU-Pro's "<image 3>"
        # options, so those items are dropped and counted
        FIGOPT = re.compile(r"^(figure|fig\.?|image|img\.?|panel|picture|photo|graph|option)\s*[A-Z0-9]{1,2}\.?$", re.I)
        n_noimg = n_badgold = n_figopt = 0
        for line in open(jp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = re.split(r"\s*Answer Choices\s*:", norm(r["question"]), maxsplit=1)[0].strip()
            od = r["options"]
            if isinstance(od, str):
                od = json.loads(od)
            letters = sorted(od)
            opts = [norm(od[L]) for L in letters]
            lab = str(r["label"]).strip().upper()[:1]
            if lab not in letters:
                n_badgold += 1
                continue
            if any(FIGOPT.match(o) for o in opts):
                n_figopt += 1
                continue
            paths = [idx[fn] for fn in (r.get("images") or []) if fn in idx]
            if not paths:
                n_noimg += 1
                continue
            cites = " ".join(f"<image {k}>" for k in range(1, len(paths) + 1))
            rows.append(dict(article="", question=f"{cites} {q}", options=opts,
                             answer=letters.index(lab),
                             _topic=str(r.get("body_system", "")), _images=paths))
        print(f"loaded {len(rows)} MedXpertQA-MM questions, grouping by body system")
        print(f"  dropped {n_figopt} with figure-valued options (not poolable), "
              f"{n_noimg} with no image file, {n_badgold} with an unreadable gold")
    elif args.dataset in VIDEO:
        # Video benchmarks. The context is the video, given to the model as a
        # fixed number of frames prepared once by prepare_video.py, which
        # writes a json index {video key: [frame paths]}. Groups are formed by
        # video, so every question in a group shares the frames.
        if not args.frames_index or not os.path.exists(args.frames_index):
            raise SystemExit("--frames-index <json from prepare_video.py> is required "
                             "for video benchmarks")
        frames = json.load(open(args.frames_index, encoding="utf-8"))
        n_novid = n_badgold = 0
        if args.dataset == "videomme":
            # lmms-lab/Video-MME: 2,700 questions, three per video, options as
            # "A. text", answer a letter, videoID is the YouTube id
            for r in load_dataset("lmms-lab/Video-MME", split="test", **rev):
                key = str(r["videoID"])
                if key not in frames:
                    n_novid += 1
                    continue
                opts = [norm(re.sub(r"^[A-D][\.\):]\s*", "", o)) for o in r["options"]]
                lab = str(r["answer"]).strip().upper()[:1]
                gi = "ABCD".find(lab)
                if not (0 <= gi < len(opts)):
                    n_badgold += 1
                    continue
                rows.append(dict(article="", question=norm(r["question"]), options=opts,
                                 answer=gi, _topic=key, _frames=frames[key],
                                 _video_id=str(r["video_id"])))
        else:
            # THUDM/LVBench video_info.meta.jsonl: {key, type, qa:[{uid, question
            # with "(A) ..." lines, answer letter, question_type, time_reference}]}
            from huggingface_hub import hf_hub_download
            mp = hf_hub_download("THUDM/LVBench", "video_info.meta.jsonl", repo_type="dataset", **rev)
            for line in open(mp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                v = json.loads(line)
                key = str(v["key"])
                if key not in frames:
                    n_novid += 1
                    continue
                for qa in v.get("qa", []):
                    text = qa["question"]
                    parts = re.split(r"\n?\s*\(([A-D])\)\s*", text)
                    # parts = [stem, 'A', textA, 'B', textB, ...]
                    stem = norm(parts[0])
                    letters, opts = parts[1::2], [norm(t) for t in parts[2::2]]
                    lab = re.sub(r"[^A-D]", "", str(qa["answer"]).upper())[:1]
                    if len(opts) < 2 or lab not in letters:
                        n_badgold += 1
                        continue
                    rows.append(dict(article="", question=stem, options=opts,
                                     answer=letters.index(lab), _topic=key,
                                     _frames=frames[key]))
        print(f"loaded {len(rows)} {args.dataset} questions from "
              f"{len({r['_topic'] for r in rows})} videos with frames, grouping by video")
        print(f"  dropped {n_novid} questions whose video has no frames, "
              f"{n_badgold} with an unreadable gold")
    elif args.dataset == "gpqa":
        # gated on the hub: accept the license and set HF_TOKEN first. Options
        # ship UNORDERED (correct + 3 incorrect fields), so a canonical
        # gold-first order would leak position into the mirror-built MCQ arm.
        # Shuffle per item with the seeded rng at load time.
        for r in load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", **rev):
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
            for r in load_dataset("ehovy/race", args.race_config, split=sp.strip(), **rev):
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
        for subj in get_dataset_config_names("ceval/ceval-exam", **rev):
            for r in load_dataset("ceval/ceval-exam", subj, split="val", **rev):
                opts = [norm(r["A"]), norm(r["B"]), norm(r["C"]), norm(r["D"])]
                gi = LETTERS.find(str(r["answer"]).strip().upper()[:1])
                if not (0 <= gi < 4):
                    continue
                rows.append(dict(article="", question=r["question"],
                                 options=opts, answer=gi, _topic=subj))
        print(f"loaded {len(rows)} C-Eval questions, grouping by subject")
    else:
        for sp in args.splits.split(","):
            rows += list(load_dataset("emozilla/quality", split=sp.strip(), **rev))
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
            f"gold_of() convention may be wrong; check the dataset's answer indexing")

    if args.drop_images:
        # images are removed at EMISSION, not here: the rows keep their image
        # lists so the global renumbering of <image k> citations runs exactly
        # as in the sighted build. The blind prompt is then the sighted prompt
        # minus the pictures and nothing else, which is what a paired control
        # requires.
        n_had = sum(1 for r in rows if r.get("_images"))
        print(f"--drop-images: {n_had} questions will be emitted without their "
              f"images, text and <image k> citations kept (closed-book screen)")

    # mmmupro is passage-free but NOT context-free: its context is the image,
    # so --exclude is meaningful there and drives the same filter the text
    # benchmarks use. The other three have no context to withhold at all.
    if args.exclude and args.dataset in CLOSED_BOOK:
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
    if args.only and os.path.exists(args.only):
        keep_h = {l.strip() for l in open(args.only, encoding="utf-8") if l.strip()}
        before = len(rows)
        rows = [r for r in rows
                if hashlib.sha1(norm(r["question"]).lower().encode()).hexdigest()[:16] in keep_h]
        print(f"--only: kept {len(rows)} of {before} questions that the screen "
              f"covered and passed")

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
                j = label_index(letter)
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
            answer = [label(pos[g]) for g in golds]
            out_groups.append(dict(
                group_id=r0["group_id"], article_key=key, n=len(qs), m=len(cand),
                arm="easy", withheld=[], n_cross_pad=len(pad), none_letter="",
                questions=qs, candidates=cand, answer=answer,
                # the mirror's questions are already globally renumbered and
                # its image list is in that order: copy both, the pool is the
                # only thing that changes in this control
                images=([] if args.drop_images else r0.get("images", [])),
                prompt=build_prompt(art_text[key], qs, cand,
                                    allow_none=False, with_passage=True)))
        with open(args.out, "w", encoding="utf-8") as f:
            for g in out_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(out_groups)} easy groups -> {args.out} "
              f"({missing} mirror groups skipped)")
        print("expectation: accuracy near the same models' MCQ accuracy. A gap "
              "here would be a cost of the format itself, since no wrong option "
              "in this pool is plausible.")
        return

    if args.cross_domain:
        # CROSS-DOMAIN GROUPS. Take every question of the mirror build and
        # regroup them so that no two questions in a group share a context.
        # Each question keeps its own full option set, so the pool size, the
        # prompt shape and the output grammar match the mirror, while the
        # neighbouring questions and their options come from unrelated
        # contexts. Compared with the same-context pool, this shows how much
        # of the drop the answers of related neighbours account for.
        if not mirror_recs:
            raise SystemExit("--cross-domain needs --mirror <same-context pooled build>")
        if args.dataset in VIDEO:
            raise SystemExit("cross-domain is not built for video: a group would need "
                             "the frames of N videos in one prompt")
        by_q = {}
        for r in rows:
            by_q.setdefault(qkey(r["question"]), r)

        def key_of(r):
            return r["_topic"] if args.dataset in PASSAGE_FREE else art_key(r["article"])

        def type_of(r):
            # numeric option sets are not separable by domain, so a group
            # prefers to mix a numeric question with textual ones
            num = sum(bool(re.fullmatch(r"[\s\d\.\-\+eE%$,/:^()x×]+", o)) for o in r["options"])
            return "num" if num >= len(r["options"]) / 2 else "text"

        items, missing = [], 0
        for r0 in mirror_recs:
            wh = set(r0.get("withheld", []))
            for qi, q in enumerate(r0.get("questions", [])):
                row = by_q.get(qkey(q))
                if row is None:
                    missing += 1
                    continue
                items.append(dict(row=row, key=key_of(row), typ=type_of(row),
                                  removed=(qi in wh), src=r0["group_id"]))
        rng.shuffle(items)
        open_groups, done = [], []
        for it in items:
            placed = False
            for strict in (True, False):        # first pass also separates option types
                for g in open_groups:
                    if any(x["key"] == it["key"] for x in g):
                        continue
                    if strict and any(x["typ"] == it["typ"] for x in g):
                        continue
                    g.append(it); placed = True
                    if len(g) == args.n:
                        done.append(g); open_groups.remove(g)
                    break
                if placed:
                    break
            if not placed:
                open_groups.append([it])
        leftover = sum(len(g) for g in open_groups)

        out_groups, skipped_dup, skipped_alpha = [], 0, 0
        for gi_, g in enumerate(done):
            chunk = [x["row"] for x in g]
            golds = [gold_of(r) for r in chunk]
            if len(set(golds)) != len(golds):
                skipped_dup += 1
                continue
            withheld_idx = {i for i, x in enumerate(g) if x["removed"]}
            if not withheld_idx and args.withhold:
                k_w = int(round(args.withhold * args.n))
                withheld_idx = set(rng.sample(range(args.n), k_w)) if k_w else set()
            present = [gd for i, gd in enumerate(golds) if i not in withheld_idx]
            cand, seen_d = list(present), set(present)
            for r in chunk:
                gd = gold_of(r)
                dis = [norm(o) for o in r["options"] if norm(o) != gd]
                if len(dis) > args.max_distractors:
                    dis = rng.sample(dis, args.max_distractors)
                for o in dis:
                    if o not in seen_d:
                        cand.append(o); seen_d.add(o)
            if len(cand) > 26:
                skipped_alpha += 1
            rng.shuffle(cand)
            pos = {c: i for i, c in enumerate(cand)}
            answer = [("none" if i in withheld_idx else label(pos[golds[i]]))
                      for i in range(args.n)]
            questions = [norm(r["question"]) for r in chunk]
            images = []
            if any(r.get("_images") for r in chunk):
                questions, images = renumber_images(
                    questions, [r.get("_images", []) for r in chunk])
            if args.dataset in PASSAGE_FREE:
                article = ""
            else:
                # one passage per question, labelled, and the question says
                # which passage it belongs to
                article = "\n\n".join(f"[Passage {i+1}]\n{norm(r['article'])}"
                                      for i, r in enumerate(chunk))
                questions = [f"[Passage {i+1}] {q}" for i, q in enumerate(questions)]
            prompt = build_prompt(article, questions, cand,
                                  allow_none=bool(withheld_idx), with_passage=True)
            out_groups.append(dict(
                group_id=f"xd_{gi_:05d}", article_key="xd", n=args.n, m=len(cand),
                arm="matching", cross_domain=True,
                source_groups=sorted({x["src"] for x in g}),
                withheld=sorted(withheld_idx), n_cross_pad=0,
                questions=questions, candidates=cand, answer=answer,
                images=([] if args.drop_images else images), prompt=prompt))
        with open(args.out, "w", encoding="utf-8") as f:
            for g in out_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        ms = sorted({g["m"] for g in out_groups})
        print(f"\nwrote {len(out_groups)} cross-domain groups -> {args.out}   "
              f"(M {ms[0]} to {ms[-1]}; {missing} mirror questions not found; "
              f"{leftover} questions left in incomplete groups; skipped {skipped_dup} "
              f"for duplicate golds{f'; {skipped_alpha} use two-letter labels' if skipped_alpha else ''})")
        n_rm = sum(len(g["withheld"]) for g in out_groups)
        if n_rm:
            print(f"  removed answers carried from the mirror: {n_rm}")
        print("expectation: accuracy near the same models' MCQ accuracy; the own-set "
              "fraction (own_set.py) should sit near 1. Compare with the same-context "
              "pool (pair_xd.py) on shared questions.")
        if out_groups:
            print("\n--- sample prompt (truncated) ---")
            p = out_groups[0]["prompt"]
            print(p[:700] + ("\n... [truncated] ...\n" + p[-400:] if len(p) > 1200 else ""))
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
            by_q.setdefault(qkey(r["question"]), r)
        out_groups, missing = [], 0
        for r0 in mirror_recs:
            for qi, q in enumerate(r0.get("questions", [])):
                row = by_q.get(qkey(q))
                if row is None:
                    missing += 1
                    continue
                # the mirror stores the POOLED question text, whose image
                # markers were renumbered across the group. A single-item arm
                # shows one question with only its own images, so take the
                # source text back. Identical to q on every text dataset.
                q = norm(row["question"])
                opts = [norm(o) for o in row["options"]]
                gold = gold_of(row)
                gi = opts.index(gold)
                blk = "\n".join(f"{label(j)}. {o}" for j, o in enumerate(opts))
                art = row.get("article", "")
                fr = row.get("_frames", [])
                if args.arm == "mcq_no_passage":
                    p = (f"Answer the question. The source passage is not "
                         f"provided; answer as best you can.\n\n"
                         f"QUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
                         f"Respond with only the letter of the correct option.")
                elif fr and args.drop_images:
                    # the no-context screen for a video benchmark
                    p = (f"Answer the question about a video. The video is not "
                         f"provided; answer as best you can.\n\n"
                         f"QUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
                         f"Respond with only the letter of the correct option.")
                elif fr:
                    p = (f"{video_note(len(fr))} Answer the question about the "
                         f"video.\n\nQUESTION: {q}\n\nOPTIONS:\n{blk}\n\n"
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
                    candidates=opts, answer=[label(gi)],
                    images=([] if args.drop_images
                            else (row.get("_images") or row.get("_frames") or [])),
                    prompt=p))
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
                blk = "\n".join(f"{label(j)}. {o}" for j, o in enumerate(cand))
                if args.arm == "mcq_none":
                    none_letter = label(cand.index(NONE_TEXT))
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
                    answer=(["none"] if bad else [label(gi)]),
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
                    j = label_index(a0[qi])
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
                # labels run A..Z then AA, AB, ... so pools above 26 are legal
                # (10-option items with all 9 distractors at N=3 give M=30).
                # Counted so the summary says when it happened: the parsers
                # switch to two-letter mode for such records.
                if len(cand) > 26:
                    skipped_alpha += 1
                rng.shuffle(cand)
                pos = {c: i for i, c in enumerate(cand)}
                answer = [("none" if i in withheld_idx else label(pos[golds[i]]))
                          for i in range(args.n)]
                questions = [norm(r["question"]) for r in chunk]
                images = []
                if any(r.get("_images") for r in chunk):
                    questions, images = renumber_images(
                        questions, [r.get("_images", []) for r in chunk])
                article = chunk[0]["article"]
                # video: the group shares one set of frames, attached once and
                # explained by a note in place of the passage block. With
                # --drop-images the note says the video is absent (the
                # no-context screen for video).
                frames = chunk[0].get("_frames", [])
                note = ""
                if frames:
                    images = [] if args.drop_images else list(frames)
                    note = (video_note(len(frames)) if not args.drop_images else
                            "The questions are about a video that is not provided; "
                            "answer as best you can.")
                prompt = build_prompt(article, questions, cand,
                                      allow_none=bool(withheld_idx),
                                      with_passage=(args.arm != "no_passage"),
                                      context_note=note)
                if args.arm == "choices_only":
                    # the slots are assigned to the questions in a random order,
                    # so slot position carries no information about which
                    # question it holds
                    if not args.keep_slot_order:
                        perm = list(range(args.n))
                        rng_slot.shuffle(perm)
                        questions = [questions[j] for j in perm]
                        answer = [answer[j] for j in perm]
                    prompt = (
                        "This is a control condition. You are shown ONLY a list of candidate "
                        "answers; the passage and the questions have been deliberately withheld.\n\n"
                        + "\n".join(f"{label(i)}. {c}" for i, c in enumerate(cand))
                        + f"\n\nThere are {args.n} hidden questions. Assign a distinct candidate "
                          f"letter to each one.\n\nYou are expected to guess — there is no way to "
                          f"know the correct assignment from this information, and that is the "
                          f"point of the control. Do not refuse, apologise, or explain.\n\n"
                          f"Output exactly {args.n} lines in the form `<number>: <letter>`, "
                          f"and nothing else.")
                groups.append(dict(
                    group_id=f"{key}_{start}", article_key=key, n=args.n, m=len(cand),
                    arm=args.arm, withheld=sorted(withheld_idx), n_cross_pad=0,
                    questions=questions, candidates=cand, answer=answer,
                    slot_shuffled=(args.arm == "choices_only" and not args.keep_slot_order),
                    # choices_only is the partial-input probe: it withholds the
                    # questions, so it must withhold their images too or the
                    # control leaks the very thing it is measuring.
                    images=([] if (args.arm == "choices_only" or args.drop_images)
                            else images),
                    prompt=prompt))
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
                answer = [("none" if i in withheld_idx else label(pos[golds[i]]))
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
                        none_letter = label(opts.index(NONE_TEXT))
                        gi = -1 if absent else opts.index(gold)
                    blk = "\n".join(f"{label(j)}. {o}" for j, o in enumerate(opts))
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
                        answer=[("none" if gi < 0 else label(gi))], prompt=p))
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
                    + "\n".join(f"{label(i)}. {c}" for i, c in enumerate(candidates))
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
          f"{f'; {skipped_alpha} use two-letter labels, pool > 26' if skipped_alpha else ''})")
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
