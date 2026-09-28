#!/usr/bin/env python
# ============================================================================
# rescore.py — re-parse the SAVED raw outputs with the current parser and report
# what changed. Zero API calls, zero GPU.
#
# Why this exists: a parser that reads the first letter after "1." cannot tell
# an answer from an echoed question. "1. Are there indications that..." yields a
# perfectly valid "A", so the run reports 0 unparsed and the wrong answer is
# indistinguishable from a real one in the accuracy column.
#
# M IS NOT OPTIONAL. m is the pool size and decides which letters are legal.
# With --distractors m=20 (A-T); guessing m=n=5 rejects every F-T answer and
# silently deletes ~75% of the data while looking like a successful rescore.
# m is therefore read from the SOURCE .jsonl (<stem>.jsonl) and a file whose m
# cannot be resolved is SKIPPED, never guessed.
#
#   python -m AnswerPooling.rescore --glob "h_*.results.jsonl"
#   python -m AnswerPooling.rescore --glob "h_*.results.jsonl" --write
# ============================================================================
import argparse
import glob
import json
import os
import shutil
from collections import Counter

from .run_matching import parse_assignment, parse_single


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


def rec_id(r):
    return r.get("item_id") or r.get("key") or r.get("group_id")


def source_m(path):
    """-> (per-id m map, modal m) from <stem>.jsonl, or (None, None)."""
    stem = os.path.basename(path).split(".")[0]
    src = os.path.join(os.path.dirname(path) or ".", stem + ".jsonl")
    if not os.path.exists(src):
        return None, None
    by_id, all_m = {}, []
    for r in load(src):
        m = r.get("m")
        if not m:
            continue
        all_m.append(m)
        by_id.setdefault(rec_id(r), m)
    if not all_m:
        return None, None
    return by_id, Counter(all_m).most_common(1)[0][0]


def acc(recs):
    ok = tot = 0
    for r in recs:
        wh = set(r.get("withheld", []))
        for i, (g, p) in enumerate(zip(r["gold"], r["pred"])):
            if i in wh or p is None:
                continue
            tot += 1
            ok += (p == g)
    return (ok / tot if tot else 0.0), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.results.jsonl")
    ap.add_argument("--write", action="store_true", help="overwrite pred (keeps .bak)")
    ap.add_argument("--force", action="store_true",
                    help="write even if coverage DROPS (it never should)")
    ap.add_argument("--show-lost", type=int, default=0, metavar="N",
                    help="dump N lines whose slot went parsed -> None. A drop means "
                         "either the OLD parser fabricated (good riddance) or the NEW "
                         "one over-rejects (a regression) -- only the raw text says which.")
    ap.add_argument("--dump-raw", type=int, default=0, metavar="N",
                    help="print the COMPLETE raw output of the first N records that lost "
                         "a slot. --show-lost guesses which line the parser read by number "
                         "prefix and that guess is wrong whenever the answers live in a "
                         "trailing block; this shows the whole response instead.")
    a = ap.parse_args()

    print(f"{'file':46s} {'old acc':>9s} {'new acc':>9s} {'old n':>7s} {'new n':>7s} {'chg':>6s}  m")
    print("-" * 96)
    skipped = []
    for path in sorted(glob.glob(a.glob)):
        recs = load(path)
        if not recs or "raw" not in recs[0]:
            continue

        by_id, modal = source_m(path)
        if modal is None:
            skipped.append((os.path.basename(path), "no source .jsonl with m"))
            continue

        old_acc, old_n = acc(recs)
        changed, ms = 0, []
        new_preds, lost, dumped = [], [], []
        for r in recs:
            arm, raw = r.get("arm", ""), r.get("raw") or ""
            n = len(r["gold"])
            m = r.get("m")
            if not m and isinstance(r.get("candidates"), list) and r["candidates"]:
                m = len(r["candidates"])
            if not m:
                m = by_id.get(rec_id(r), modal)
            ms.append(m)
            new = ([parse_single(raw, m)] if arm.startswith("mcq")
                   else parse_assignment(raw, n, m))
            if len(new) != len(r["pred"]):
                new = r["pred"]
            changed += sum(1 for x, y in zip(new, r["pred"]) if x != y)
            if (a.dump_raw and len(dumped) < a.dump_raw
                    and any(o is not None and p is None
                            for o, p in zip(r["pred"], new))):
                dumped.append((r["pred"], new, r["gold"], m, raw))
            if a.show_lost and len(lost) < a.show_lost:
                for i, (old_p, new_p) in enumerate(zip(r["pred"], new)):
                    if old_p is not None and new_p is None:
                        line = next((l.strip() for l in raw.splitlines()
                                     if l.strip().startswith(str(i + 1))), "<no line>")
                        gold = r["gold"][i] if i < len(r["gold"]) else "?"
                        lost.append((old_p, gold, line[:190]))
                        if len(lost) >= a.show_lost:
                            break
            new_preds.append(new)

        saved = [r["pred"] for r in recs]
        for r, p in zip(recs, new_preds):
            r["pred"] = p
        new_acc, new_n = acc(recs)

        mset = sorted(set(ms))
        flag = ""
        if new_n < old_n:
            flag = "  <-- COVERAGE DROPPED, not written"
        elif changed and abs(new_acc - old_acc) > 0.005:
            flag = "  <-- rescored"
        print(f"{os.path.basename(path):46s} {old_acc:9.4f} {new_acc:9.4f} "
              f"{old_n:7d} {new_n:7d} {changed:6d}  "
              f"{mset[0] if len(mset) == 1 else mset}{flag}")
        for old_p, gold, line in lost:
            hit = "CORRECT" if old_p == gold else "wrong  "
            print(f"      lost old={old_p} gold={gold} ({hit})  {line!r}")
        for k, (old_p, new_p, gold, m, raw) in enumerate(dumped, 1):
            print(f"\n{'='*88}\nRECORD {k}   m={m} (letters A-{chr(64+m)})")
            print(f"  old pred: {old_p}\n  new pred: {new_p}\n  gold    : {gold}")
            print(f"{'-'*88}\n{raw}\n{'='*88}")

        # A correct reparse can only ADD parsed slots. Losing coverage means the
        # parser or m is wrong, so refuse to persist it.
        if new_n < old_n and not a.force:
            for r, p in zip(recs, saved):
                r["pred"] = p
            continue
        if a.write and changed:
            shutil.copyfile(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name, why in skipped:
        print(f"SKIPPED {name}: {why}")
    print("\nwritten (.bak kept). re-run compare.py" if a.write
          else "\ndry run — add --write to apply")


if __name__ == "__main__":
    main()
