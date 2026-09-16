#!/usr/bin/env python
# ============================================================================
# bench_datasets.py — one loader per benchmark, returning a UNIFORM row format
# that build_matching.py / own_set.py can pool without knowing the source.
#
# Every loader yields dicts with:
#   article   str        shared passage ("" for passage-free benchmarks)
#   question  str        the question text (image refs rewritten to [image k])
#   options   [str]      the answer options, in the dataset's original order
#   answer    int        0-BASED index of the gold option
#   _topic    str        grouping key for passage-free benchmarks
#   _qid      str        stable per-item id (join key for mirror-built arms;
#                        question TEXT is not unique on visual benchmarks,
#                        e.g. MMMU-Pro vision where the text is "(see image)")
#   _images   [PIL]      zero or more images that belong to the question
#
# Text-only benchmarks (quality, race, mmlupro, gpqa, ceval) keep living in
# build_matching.py; this module adds the rest and re-exports GPQA so that a
# --gpqa-config switch (diamond / main / extended) exists in one place.
#
# Visual benchmarks (MMMU, MMMU-Pro, MathVista, ScienceQA) attach images to
# the row. build_matching.py writes them to <out>_images/ and records their
# paths; run_matching.py sends them alongside the prompt.
# ============================================================================
import ast
import hashlib
import io
import os
import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# which benchmarks need a multimodal model
VISUAL = {"mmmu", "mmmu_pro", "mathvista", "scienceqa"}
# all benchmarks added by this module (passage-free: they group by topic)
NEW = {"gpqa_diamond", "hellaswag", "mmmu", "mmmu_pro", "mathvista", "scienceqa"}

# dataset-specific header line for the pooled prompt. The default header says
# "match each question to its correct answer"; HellaSwag has no question, it
# has a context and four continuations, so the wording must say so or the
# model looks for a question mark that never arrives.
TASK_TEXT = {
    "hellaswag": ("Each numbered item below is the beginning of a short passage. "
                  "Match each one to the candidate that most plausibly continues it."),
}


def norm(s):
    return " ".join(str(s).split()).strip()


def _sha(s):
    return hashlib.sha1(norm(s).encode("utf-8")).hexdigest()[:16]


def _to_pil(img):
    """datasets returns PIL images already; guard against bytes / dict."""
    if img is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("pip install pillow")
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and img.get("bytes"):
        return Image.open(io.BytesIO(img["bytes"]))
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img))
    return None


def _parse_options(raw):
    """MMMU ships options as the STRING repr of a python list ("['A', 'B']").
    MMMU-Pro too. Fall back to a real list or a single string."""
    if isinstance(raw, (list, tuple)):
        return [norm(o) for o in raw]
    s = str(raw).strip()
    if s.startswith("["):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [norm(o) for o in v]
        except Exception:
            pass
    return [norm(s)]


def _letter_index(a, k):
    s = str(a).strip().upper()
    if s and s[0] in LETTERS[:k] and (len(s) == 1 or not s[1].isalnum()):
        return LETTERS.index(s[0])
    return None


_IMG_TAG = re.compile(r"<image\s*(\d+)>", re.I)


# ------------------------------------------------------------------- GPQA --
def load_gpqa(rng, config="gpqa_diamond"):
    """Idavidrein/gpqa, gated: accept the licence and export HF_TOKEN.
    Options ship UNORDERED (correct + 3 incorrect fields), so shuffle per item
    with the seeded rng, exactly as build_matching.py does for gpqa_main."""
    from datasets import load_dataset
    rows = []
    for r in load_dataset("Idavidrein/gpqa", config, split="train"):
        opts = [norm(r["Correct Answer"]), norm(r["Incorrect Answer 1"]),
                norm(r["Incorrect Answer 2"]), norm(r["Incorrect Answer 3"])]
        gold = opts[0]
        rng.shuffle(opts)
        rows.append(dict(article="", question=norm(r["Question"]), options=opts,
                         answer=opts.index(gold), _topic=str(r["Subdomain"]),
                         _qid=f"gpqa:{_sha(r['Question'])}", _images=[]))
    print(f"loaded {len(rows)} GPQA questions ({config}), grouping by subdomain")
    return rows


# -------------------------------------------------------------- HellaSwag --
def load_hellaswag(rng, splits="validation"):
    """Rowan/hellaswag. The 'question' is the context (ctx) and the options
    are the 4 endings; label is a 0-based index as a string. Test split has
    no labels and is skipped. Groups by activity_label so pooled endings come
    from the SAME activity, which is what makes them confusable."""
    from datasets import load_dataset
    rows = []
    for sp in splits.split(","):
        sp = sp.strip()
        if not sp:
            continue
        for r in load_dataset("Rowan/hellaswag", split=sp):
            lab = str(r.get("label", "")).strip()
            if not lab.isdigit():
                continue
            ends = [norm(e) for e in r["endings"]]
            gi = int(lab)
            if not (0 <= gi < len(ends)):
                continue
            rows.append(dict(article="", question=norm(r["ctx"]), options=ends,
                             answer=gi, _topic=str(r["activity_label"]),
                             _qid=f"hellaswag:{r.get('ind', _sha(r['ctx']))}",
                             _images=[]))
    print(f"loaded {len(rows)} HellaSwag items ({splits}), grouping by activity")
    return rows


# ------------------------------------------------------------------- MMMU --
def _collect_images(r, max_k=7):
    imgs = []
    for k in range(1, max_k + 1):
        im = _to_pil(r.get(f"image_{k}"))
        if im is not None:
            imgs.append((k, im))
    return imgs


def _rewrite_image_tags(text, present):
    """'<image 1>' -> '[image 1]' so the placeholder survives norm() and the
    model can tie the attached image back to the question. 'present' is the
    set of image numbers actually attached; a tag with no image is left as is
    (and the item is dropped upstream)."""
    return _IMG_TAG.sub(lambda m: f"[image {m.group(1)}]", text)


def load_mmmu(rng, split="validation", subjects=""):
    """MMMU/MMMU_Pro is the harder sibling; this is MMMU/MMMU (30 subject
    configs). validation has answers, test does not. Only 'multiple-choice'
    items are usable: open items have no options to pool.

    Dropped: items whose OPTIONS reference an image (the candidate would be a
    picture, and the pool is text), and items that reference an image number
    with no image attached."""
    from datasets import load_dataset, get_dataset_config_names
    names = [s for s in subjects.split(",") if s.strip()] or \
        get_dataset_config_names("MMMU/MMMU")
    rows, dropped = [], 0
    for subj in names:
        subj = subj.strip()
        for r in load_dataset("MMMU/MMMU", subj, split=split):
            if str(r.get("question_type", "")).lower() != "multiple-choice":
                continue
            opts = _parse_options(r["options"])
            if any(_IMG_TAG.search(o) for o in opts) or len(opts) < 2:
                dropped += 1
                continue
            gi = _letter_index(r["answer"], len(opts))
            if gi is None:
                dropped += 1
                continue
            imgs = _collect_images(r)
            need = {int(m) for m in _IMG_TAG.findall(r["question"])}
            have = {k for k, _ in imgs}
            if need - have:
                dropped += 1
                continue
            q = _rewrite_image_tags(r["question"], have)
            rows.append(dict(article="", question=norm(q), options=opts, answer=gi,
                             _topic=subj, _qid=f"mmmu:{r['id']}",
                             _images=[im for _, im in imgs]))
    print(f"loaded {len(rows)} MMMU multiple-choice items ({split}), "
          f"{dropped} dropped (image options / missing image / bad answer), "
          f"grouping by subject")
    return rows


def load_mmmu_pro(rng, config="standard (4 options)"):
    """MMMU/MMMU_Pro. Configs: 'standard (4 options)', 'standard (10 options)',
    'vision'. In 'vision' the question AND options are printed inside the
    image, so the pooled prompt can only show '[see image]' as the question;
    the options list still ships as text, which is what gets pooled."""
    from datasets import load_dataset
    rows, dropped = [], 0
    vision = config.strip().lower() == "vision"
    for r in load_dataset("MMMU/MMMU_Pro", config, split="test"):
        opts = _parse_options(r["options"])
        if any(_IMG_TAG.search(o) for o in opts) or len(opts) < 2:
            dropped += 1
            continue
        gi = _letter_index(r["answer"], len(opts))
        if gi is None:
            dropped += 1
            continue
        if vision:
            im = _to_pil(r.get("image"))
            if im is None:
                dropped += 1
                continue
            imgs, q = [im], "(The question is printed in the attached image.)"
        else:
            pairs = _collect_images(r)
            need = {int(m) for m in _IMG_TAG.findall(r["question"])}
            have = {k for k, _ in pairs}
            if need - have:
                dropped += 1
                continue
            imgs = [im for _, im in pairs]
            q = _rewrite_image_tags(r["question"], have)
        rows.append(dict(article="", question=norm(q), options=opts, answer=gi,
                         _topic=str(r.get("subject", "")),
                         _qid=f"mmmupro:{r['id']}", _images=imgs))
    print(f"loaded {len(rows)} MMMU-Pro items ({config}), {dropped} dropped, "
          f"grouping by subject")
    return rows


# -------------------------------------------------------------- MathVista --
def load_mathvista(rng, split="testmini", group_by="task"):
    """AI4Math/MathVista. Only question_type == 'multi_choice' items carry
    options; free_form items need a judge and are skipped. Gold is the answer
    TEXT, matched back to its choice. group_by: task | category | skill |
    source | grade (all live in metadata)."""
    from datasets import load_dataset
    rows, dropped = [], 0
    for r in load_dataset("AI4Math/MathVista", split=split):
        if str(r.get("question_type", "")) != "multi_choice":
            continue
        opts = [norm(c) for c in (r.get("choices") or [])]
        if len(opts) < 2:
            dropped += 1
            continue
        gold = norm(r["answer"])
        if gold not in opts:
            dropped += 1
            continue
        im = _to_pil(r.get("decoded_image"))
        if im is None:
            dropped += 1
            continue
        md = r.get("metadata") or {}
        key = md.get(group_by, "")
        if isinstance(key, list):
            key = ",".join(map(str, key))
        rows.append(dict(article="", question=norm(r["question"]), options=opts,
                         answer=opts.index(gold), _topic=str(key or "unknown"),
                         _qid=f"mathvista:{r['pid']}", _images=[im]))
    print(f"loaded {len(rows)} MathVista multi-choice items ({split}), "
          f"{dropped} dropped, grouping by {group_by}")
    return rows


# -------------------------------------------------------------- ScienceQA --
def load_scienceqa(rng, splits="test", group_by="topic", require_image=False):
    """derek-thomas/ScienceQA. answer is a 0-based index. 'hint' is a short
    context that the official prompt includes, so it is prepended to the
    question. About half the items have an image; --require-image keeps only
    those (the IMG subset reported in papers)."""
    from datasets import load_dataset
    rows, dropped = [], 0
    for sp in splits.split(","):
        sp = sp.strip()
        if not sp:
            continue
        for i, r in enumerate(load_dataset("derek-thomas/ScienceQA", split=sp)):
            opts = [norm(c) for c in r["choices"]]
            gi = int(r["answer"])
            if not (0 <= gi < len(opts)) or len(opts) < 2:
                dropped += 1
                continue
            im = _to_pil(r.get("image"))
            if require_image and im is None:
                dropped += 1
                continue
            hint = norm(r.get("hint") or "")
            q = (f"Context: {hint} Question: {norm(r['question'])}"
                 if hint else norm(r["question"]))
            rows.append(dict(article="", question=q, options=opts, answer=gi,
                             _topic=str(r.get(group_by, "")),
                             _qid=f"scienceqa:{sp}:{i}",
                             _images=([im] if im is not None else [])))
    print(f"loaded {len(rows)} ScienceQA items ({splits}"
          f"{', image-only' if require_image else ''}), {dropped} dropped, "
          f"grouping by {group_by}")
    return rows


# ---------------------------------------------------------------- dispatch --
def load_rows(dataset, rng, args):
    """Entry point used by build_matching.py and own_set.py."""
    if dataset == "gpqa_diamond":
        return load_gpqa(rng, "gpqa_diamond")
    if dataset == "gpqa":
        return load_gpqa(rng, getattr(args, "gpqa_config", "gpqa_main"))
    if dataset == "hellaswag":
        return load_hellaswag(rng, getattr(args, "splits", "validation") or "validation")
    if dataset == "mmmu":
        return load_mmmu(rng, getattr(args, "mmmu_split", "validation"),
                         getattr(args, "subjects", ""))
    if dataset == "mmmu_pro":
        return load_mmmu_pro(rng, getattr(args, "mmmu_pro_config", "standard (4 options)"))
    if dataset == "mathvista":
        return load_mathvista(rng, getattr(args, "mathvista_split", "testmini"),
                              getattr(args, "group_by_visual", "task"))
    if dataset == "scienceqa":
        return load_scienceqa(rng, getattr(args, "splits", "test") or "test",
                              getattr(args, "group_by_visual", "topic"),
                              getattr(args, "require_image", False))
    raise SystemExit(f"unknown dataset {dataset}")


def save_images(images, out_dir, qid):
    """Write a row's PIL images to disk (PNG) and return their paths, relative
    to the cwd, so a .jsonl of groups stays small and diffable. Idempotent:
    the file name is derived from the item id, so rebuilding overwrites."""
    if not images:
        return []
    os.makedirs(out_dir, exist_ok=True)
    stem = hashlib.sha1(qid.encode("utf-8")).hexdigest()[:16]
    paths = []
    for k, im in enumerate(images, 1):
        p = os.path.join(out_dir, f"{stem}_{k}.png")
        if not os.path.exists(p):
            im.convert("RGB").save(p, format="PNG")
        paths.append(p)
    return paths
