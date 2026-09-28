#!/usr/bin/env python
# ============================================================================
# manifest.py -- the released pooled benchmarks, without benchmark text.
#
# A manifest describes one build record by record: identifiers of its
# questions, options, and passages, the correct letters, the removed set, a
# hash of the exact prompt, and the prompt's template, which is the prompt
# with every passage, question, and option replaced by a numbered slot. It
# holds no question, option, or passage text, so it can be shared for every
# benchmark, gated ones included, and it puts no test item on the web.
#
#   export    builds -> manifests                                  (authors)
#   rebuild   manifests + one universe build per benchmark -> builds  (anyone)
#   verify    manifests + builds -> report                            (anyone)
#
# A universe build holds every question of a benchmark once, with its options
# and its context: build_matching with --n 1. rebuild fills every template
# from it, so each released build comes back byte-identical without the
# flags, filters, or builder version that made it. A rebuilt prompt whose
# hash differs from the manifest is reported and not written.
#
#   python -m AnswerPooling.manifest export  --dir . --out-dir manifests
#   python -m AnswerPooling.manifest rebuild --universe "universe/*.jsonl" --out-dir rebuilt
#   python -m AnswerPooling.manifest verify  --dir rebuilt
#
# This file imports nothing from the package, so it also runs on its own
# (python manifest.py ...) in a folder of existing builds.
# ============================================================================
import argparse
import datetime
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter

FORMAT = "answer-pooling-manifest-v2"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMGREF = re.compile(r"<\s*image\s*(\d+)\s*>", re.I)
PASSAGE_TAG = re.compile(r"^\[Passage \d+\]\s*")      # cross-domain question prefix
PASSAGE_HEAD = re.compile(r"^\[Passage \d+\]$")        # cross-domain passage header
QLINE = re.compile(r"^(\d+\. |QUESTION: )(.*)$")
NONE_TEXT = "None of these answers is correct."        # the printed none option
SLOT = "⦃{}{}⦄"                              # e.g. ⦃Q0⦄
SLOT_RE = re.compile("⦃([PQC])(\\d+)⦄")

# file prefix -> (build_matching --dataset, Hugging Face dataset)
BENCH = {
    "h": ("quality", "emozilla/quality"),
    "wh40": ("quality", "emozilla/quality"),     # QuALITY ablations wh40_alt, wh40_reuse
    "ra": ("race", "ehovy/race"),
    "mp9": ("mmlupro", "TIGER-Lab/MMLU-Pro"),
    "gp": ("gpqa", "Idavidrein/gpqa"),
    "ce": ("ceval", "ceval/ceval-exam"),
    "mm": ("mmmupro", "MMMU/MMMU_Pro"),
    "mx": ("medxpertqa", "TsinghuaC3I/MedXpertQA"),
    "vm": ("videomme", "lmms-lab/Video-MME"),
}

# the builds the paper reports. export skips every other file in the folder.
PAPER_STEMS = (
    "h_match h_mcq h_np h_mcq_np h_easy h_xd h_wh20 h_wh40 h_wh80 h_mcqnone40 "
    "h_prose40 h_match_seed1 h_match_reuse h_match_n3 h_match_n4 h_match_n5 "
    "h_match_n6 wh40_alt wh40_reuse "
    "ra_match ra_mcq ra_np ra_mcq_np ra_wh50 ra_mcq_none ra_mcq_prose ra_xd "
    "mp9_match mp9_mcq mp9_wh40 "
    "gp_match gp_mcq gp_wh40 gp_xd "
    "ce_match ce_mcq ce_wh40 "
    "mm_match mm_mcq mm_noimg mm_np mm_easy "
    "mx_match mx_mcq mx_noimg mx_np mx_wh40 "
    "vm_match vm_mcq vm_noimg vm_np"
).split()

# screen outputs a build depends on. Both hold hashes and ids, no text.
EXTRAS = ["blind_solvable.txt", "flagged_groups.txt"]

# record fields copied as they are (all ids, letters, or numbers)
META = ("group_id", "article_key", "n", "m", "arm", "withheld", "n_cross_pad",
        "none_letter", "answer", "item_id", "q_index", "source_groups")


# ------------------------------------------------------------------ text ids --
def norm(s):
    return " ".join(str(s).split()).strip()


def sha16(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def label(i):
    """Option label i, the same as run_matching.label: A..Z, then AA, AB, ..."""
    return LETTERS[i] if i < 26 else LETTERS[(i - 26) // 26] + LETTERS[(i - 26) % 26]


def qbase(q):
    """A question without its cross-domain passage prefix."""
    return norm(PASSAGE_TAG.sub("", norm(q)))


def qid(q):
    """Question identifier: image citations canonicalised, then hashed."""
    return sha16(IMGREF.sub("<image>", qbase(q)))


def oid(o):
    return sha16(norm(o))


def phash(prompt):
    """Same fingerprint run_matching stores with every result."""
    return hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()[:16]


def tail(path):
    """Last two path components, enough to find a file and never a home dir."""
    parts = str(path).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:])


# ------------------------------------------------------------ templates --
def passage_lines(prompt):
    """(index, line) of every passage in a prompt: the line after 'PASSAGE:'
    or after a cross-domain '[Passage k]' header."""
    lines, out = prompt.split("\n"), []
    for i in range(1, len(lines)):
        prev = lines[i - 1]
        if lines[i] and (prev == "PASSAGE:" or PASSAGE_HEAD.match(prev)):
            out.append((i, lines[i]))
    return out


def template(r):
    """The prompt with every passage, question, and option replaced by a
    slot, and the passage ids in slot order. Raises if any of the record's
    own text would stay in the template, and checks that filling the template
    with that text gives the prompt back."""
    qs, cs = r.get("questions", []), r.get("candidates", [])
    qpos = {}
    for i, q in enumerate(qs):
        qpos.setdefault(q, i)
    cpos = {f"{label(j)}. {c}": j for j, c in enumerate(cs)}
    lines = r["prompt"].split("\n")
    passages = dict(passage_lines(r["prompt"]))
    out, pids = [], []
    for i, line in enumerate(lines):
        if i in passages:
            out.append(SLOT.format("P", len(pids)))
            pids.append(sha16(line))
        elif line in cpos:
            j = cpos[line]
            out.append(f"{label(j)}. " + SLOT.format("C", j))
        else:
            mo = QLINE.match(line)
            if mo and mo.group(2) in qpos:
                out.append(mo.group(1) + SLOT.format("Q", qpos[mo.group(2)]))
            else:
                out.append(line)
    tpl = "\n".join(out)
    for s in list(qs) + list(cs):
        t = qbase(s)
        if len(t) >= 12 and t in tpl:
            raise ValueError(f"text left in the template: {t[:40]!r}")
    if any(len(x) > 400 for x in out):
        raise ValueError("a long line is left in the template")
    texts = [line for _, line in sorted(passages.items())]
    if fill(tpl, texts, qs, cs) != r["prompt"]:
        raise ValueError("the template does not give the prompt back")
    return tpl, pids


def fill(tpl, P, Q, C):
    table = {"P": P, "Q": Q, "C": C}
    return SLOT_RE.sub(lambda mo: table[mo.group(1)][int(mo.group(2))], tpl)


def with_marks(base, marks):
    """Put a question's own image citations, as they appear in the build,
    back into the universe text."""
    found = IMGREF.findall(base)
    if len(found) != len(marks):
        raise KeyError("image citations differ")
    it = iter(marks)
    return IMGREF.sub(lambda mo: next(it), base)


# ------------------------------------------------------------------- io --
def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def is_build(rec):
    return isinstance(rec, dict) and "prompt" in rec and "answer" in rec and "pred" not in rec


def read_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        return header, [json.loads(line) for line in f if line.strip()]


def key(e):
    return (e["group_id"], e.get("item_id") or e.get("q_index"))


# --------------------------------------------------------------- export --
def entry(r, templates):
    tpl, pids = template(r)
    tid = sha16(tpl)[:12]
    templates[tid] = tpl
    qs = r.get("questions", [])
    e = {k: r[k] for k in META if r.get(k) is not None and r.get(k) != ""}
    e.update(q=[qid(x) for x in qs], c=[oid(x) for x in r.get("candidates", [])],
             p=pids, t=tid, phash=phash(r["prompt"]))
    tags = [(PASSAGE_TAG.match(norm(x)).group(0) if PASSAGE_TAG.match(norm(x)) else "")
            for x in qs]
    if any(tags):
        e["qp"] = tags
    marks = [[mo.group(0) for mo in IMGREF.finditer(x)] for x in qs]
    if any(marks):
        e["qm"] = marks
    if r.get("cross_domain"):
        e["cross_domain"] = True
    imgs = r.get("images") or []
    if imgs:
        e["img"] = [tail(p) for p in imgs]
    return e


def hub_revision(hub, lookup):
    """(sha, source): the revision in the local Hub cache when exactly one is
    cached, else the current revision on the Hub, else (None, reason)."""
    if not lookup:
        return None, "not looked up"
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_type == "dataset" and repo.repo_id == hub:
                revs = sorted(r.commit_hash for r in repo.revisions)
                if len(revs) == 1:
                    return revs[0], "local cache"
                if revs:
                    return None, "several cached revisions: " + ",".join(revs)
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(hub).sha, "Hub at export time"
    except Exception as ex:
        return None, f"unavailable ({type(ex).__name__})"


def export(a):
    os.makedirs(a.out_dir, exist_ok=True)
    stems = list(PAPER_STEMS) + [s.strip() for s in a.extra_stems.split(",") if s.strip()]
    revs, done, missing, failed = {}, [], [], []
    lines_seen = Counter()
    for stem in stems:
        path = os.path.join(a.dir, stem + ".jsonl")
        if not os.path.exists(path):
            missing.append(stem)
            continue
        recs = load(path)
        if not recs or not is_build(recs[0]):
            print(f"  skip {stem}: not a build file")
            continue
        templates, entries = {}, []
        try:
            for r in recs:
                entries.append(entry(r, templates))
        except ValueError as ex:
            failed.append((stem, str(ex)))
            continue
        for tpl in templates.values():
            lines_seen.update(x for x in tpl.split("\n") if x and not SLOT_RE.search(x))
        dataset, hub = BENCH.get(stem.split("_")[0], (None, None))
        if hub and hub not in revs:
            revs[hub] = hub_revision(hub, not a.no_revision)
        sha, source = revs.get(hub, (None, "unknown dataset"))
        header = dict(format=FORMAT, stem=stem, dataset=dataset, hub_dataset=hub,
                      hub_revision=sha, revision_source=source, records=len(recs),
                      created=datetime.date.today().isoformat(), templates=templates)
        out = os.path.join(a.out_dir, stem + ".manifest.jsonl.gz")
        with gzip.open(out, "wt", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        done.append((stem, len(recs), len(templates), os.path.getsize(out)))
    for name in EXTRAS:
        src = os.path.join(a.dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out_dir, name))
            print(f"  copied {name}")
    print(f"\nwrote {len(done)} manifests to {a.out_dir}")
    for stem, n, nt, size in done:
        print(f"  {stem:18s} {n:6d} records  {nt:3d} templates  {size/1024:8.1f} KB")
    for hub, (sha, source) in sorted(revs.items()):
        print(f"  revision {hub:26s} {sha or '--'}  ({source})")
    if missing:
        print(f"not found (not built, or under another name): {' '.join(missing)}")
    for stem, why in failed:
        print(f"NOT EXPORTED {stem}: {why}")
    # every line the templates keep, for a one-time check that it is all
    # instruction text and no benchmark text
    print(f"\n{len(lines_seen)} distinct template lines:")
    for line, _ in sorted(lines_seen.items()):
        print(f"  | {line}")


# -------------------------------------------------------------- rebuild --
def universe(paths):
    """Text by identifier, from one or more universe builds."""
    P, Q, C, IMG = {}, {}, {oid(NONE_TEXT): NONE_TEXT}, {}
    for path in paths:
        for r in load(path):
            if not is_build(r):
                continue
            for q in r.get("questions", []):
                Q.setdefault(qid(q), qbase(q))
            for c in r.get("candidates", []):
                C.setdefault(oid(c), norm(c))
            for _, line in passage_lines(r["prompt"]):
                P.setdefault(sha16(line), line)
            for p in r.get("images") or []:
                IMG.setdefault(tail(p), p)
    return P, Q, C, IMG


def rebuild(a):
    paths = sorted(glob.glob(a.universe))
    if not paths:
        raise SystemExit(f"no universe builds match {a.universe}")
    P, Q, C, IMG = universe(paths)
    print(f"universe: {len(Q)} questions, {len(C)} options, {len(P)} passages, "
          f"{len(IMG)} images from {len(paths)} files")
    os.makedirs(a.out_dir, exist_ok=True)
    n_bad = 0
    for mp in sorted(glob.glob(a.manifests)):
        header, entries = read_manifest(mp)
        stem, templates = header["stem"], header.get("templates", {})
        out, why = [], Counter()
        for e in entries:
            try:
                qs = [(e.get("qp") or [""] * len(e["q"]))[i]
                      + with_marks(Q[x], (e.get("qm") or [[]] * len(e["q"]))[i])
                      for i, x in enumerate(e["q"])]
                cs = [C[x] for x in e["c"]]
                ps = [P[x] for x in e["p"]]
                imgs = [IMG[x] for x in e.get("img", [])]
            except KeyError:
                why["text not in the universe"] += 1
                continue
            prompt = fill(templates[e["t"]], ps, qs, cs)
            if phash(prompt) != e["phash"]:
                why["prompt differs"] += 1
                continue
            r = {k: e[k] for k in META if k in e}
            r.update(questions=qs, candidates=cs, images=imgs, prompt=prompt)
            if e.get("cross_domain"):
                r["cross_domain"] = True
            out.append(r)
        with open(os.path.join(a.out_dir, stem + ".jsonl"), "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = len(out) == len(entries)
        n_bad += not ok
        detail = "" if ok else "   " + ", ".join(f"{k} {v}" for k, v in why.items())
        print(f"{'OK     ' if ok else 'PARTIAL'} {stem:18s} {len(out)}/{len(entries)} rebuilt{detail}")
    sys.exit(1 if n_bad else 0)


# --------------------------------------------------------------- verify --
def verify(a):
    paths = sorted(glob.glob(a.manifests))
    if not paths:
        raise SystemExit(f"no manifests match {a.manifests}")
    n_bad = 0
    for mp in paths:
        header, want = read_manifest(mp)
        stem = header["stem"]
        bp = os.path.join(a.dir, stem + ".jsonl")
        if not os.path.exists(bp):
            print(f"MISSING {stem:18s} no build at {bp}")
            n_bad += 1
            continue
        have = {}
        for r in load(bp):
            if is_build(r):
                e = dict(group_id=r["group_id"], item_id=r.get("item_id"),
                         q_index=r.get("q_index"), withheld=r.get("withheld", []),
                         answer=r.get("answer", []), phash=phash(r["prompt"]),
                         q=[qid(x) for x in r.get("questions", [])],
                         c=[oid(x) for x in r.get("candidates", [])])
                have[key(e)] = e
        same = prompt_only = items = absent = 0
        for w in want:
            h = have.pop(key(w), None)
            if h is None:
                absent += 1
            elif all(h[k] == w.get(k, []) for k in ("q", "c", "answer", "withheld")):
                if h["phash"] == w["phash"]:
                    same += 1
                else:
                    prompt_only += 1
            else:
                items += 1
        ok = same == len(want) and not have
        detail = "" if ok else (f"   different items {items}, same items but other prompt "
                                f"{prompt_only}, missing {absent}, extra {len(have)}")
        print(f"{'OK     ' if ok else 'DIFFERS'} {stem:18s} {same}/{len(want)} identical{detail}")
        n_bad += not ok
    print(f"\n{len(paths) - n_bad} of {len(paths)} builds identical to the release")
    sys.exit(1 if n_bad else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write a manifest for every paper build in --dir")
    e.add_argument("--dir", default=".", help="folder holding the build files")
    e.add_argument("--out-dir", default="manifests")
    e.add_argument("--extra-stems", default="",
                   help="comma-separated build stems to export besides the paper's")
    e.add_argument("--no-revision", action="store_true",
                   help="do not look up the dataset revisions")
    b = sub.add_parser("rebuild", help="rebuild the released builds from universe builds")
    b.add_argument("--manifests", default="manifests/*.manifest.jsonl.gz")
    b.add_argument("--universe", required=True,
                   help="glob of universe builds (build_matching --n 1), one per benchmark")
    b.add_argument("--out-dir", default="rebuilt")
    v = sub.add_parser("verify", help="check builds against the manifests")
    v.add_argument("--manifests", default="manifests/*.manifest.jsonl.gz")
    v.add_argument("--dir", default=".", help="folder holding the builds to check")
    a = ap.parse_args()
    {"export": export, "rebuild": rebuild, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    main()
