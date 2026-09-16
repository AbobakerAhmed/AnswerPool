#!/usr/bin/env python
# ============================================================================
# probe_api.py — find out why an arm returned "api errors N" before spending
# another half hour on retries.
#
# Sends three escalating requests and reports the provider's own message:
#   1. text only, tiny            -> is the key valid and funded?
#   2. text only, a real prompt   -> is the model id right, does it answer?
#   3. one image + prompt         -> does this endpoint accept images at all,
#                                    in the format we send them?
#
#   python probe_api.py --model kimi-k3
#   python probe_api.py --model opus-5 --in runs/scienceqa_match.jsonl
# ============================================================================
import argparse
import json
import os
import sys

import models
import run_matching as R


def show(tag, txt, err):
    if err:
        print(f"  {tag}: FAILED")
        print(f"     {err[:400]}")
    else:
        one = " ".join((txt or "").split())[:120]
        print(f"  {tag}: ok -> {one!r}" if one else
              f"  {tag}: ok but EMPTY text (budget exhausted? raise --max-out)")
    return not err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a models.py key or raw id")
    ap.add_argument("--in", dest="inp", default="",
                    help="a built .jsonl; the image test uses its first record "
                         "with images, so the probe matches the real payload")
    ap.add_argument("--max-out", type=int, default=512)
    a = ap.parse_args()

    r = models.resolve(a.model)
    print(f"model   {r['model']}\nbackend {r['backend']}")
    if r["base_url"]:
        os.environ["OPENAI_BASE_URL"] = r["base_url"]
        print(f"base    {r['base_url']}")
    if r["key_env"]:
        print(f"key     {r['key_env']} = "
              f"{'set' if os.environ.get(r['key_env']) else 'NOT SET'}")
    if r["backend"] == "vllm":
        raise SystemExit("vllm runs locally; nothing to probe")

    if r["backend"] == "anthropic":
        client, fn = R.make_anthropic_client(), R.call_anthropic
    elif r["backend"] == "openai":
        client, fn = R.make_openai_client(), R.call_openai
    else:
        client, fn = R.make_client(), R.call
    print()

    rec = dict(prompt="Reply with the single word: OK", images=[], n=1, m=1,
               arm="probe", answer=["A"], candidates=["x"], questions=["q"])
    if not show("1 text, tiny ", *fn(client, r["model"], rec, a.max_out, 0)):
        print("\n-> the key or the model id is the problem; the message above "
              "says which.\n   401/403 authentication, 402/insufficient "
              "balance, 404 unknown model id.")
        return

    rec2 = dict(rec, prompt=("Match each question to its answer.\n\n"
                             "QUESTIONS:\n1. What is 2+2?\n2. Capital of France?"
                             "\n\nCANDIDATE ANSWERS:\nA. Paris\nB. 4\n\n"
                             "Respond with one line per question in the form "
                             "`<question number>: <letter>`, and nothing else."))
    show("2 text, real ", *fn(client, r["model"], rec2, a.max_out, 0))

    # image test: reuse a real record so the payload is byte-for-byte what the
    # run sends, including the number of images per request
    img_rec = None
    if a.inp and os.path.exists(a.inp):
        for line in open(a.inp, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                if any(x.get("images") or []):
                    img_rec = x
                    break
    if img_rec is None:
        for cand in ("runs", "."):
            for f in sorted(os.listdir(cand)) if os.path.isdir(cand) else []:
                if f.endswith(".jsonl") and img_rec is None:
                    for line in open(os.path.join(cand, f), encoding="utf-8"):
                        if line.strip():
                            x = json.loads(line)
                            if any(x.get("images") or []):
                                img_rec = x
                                break
            if img_rec:
                break
    if img_rec is None:
        print("  3 image      : skipped (no built file with images found; "
              "pass --in <built .jsonl>)")
        return
    n_img = sum(len(x) for x in img_rec["images"])
    print(f"  (image test uses a real record: {n_img} image(s), "
          f"{img_rec['n']} question(s))")
    ok = show("3 with image ", *fn(client, r["model"], img_rec, a.max_out, 0))
    if not ok:
        print("\n-> text works but images do not. Either this model id is not "
              "the vision one,\n   or the endpoint rejects the image format. "
              "Run the text-only arms\n   (choices_only, and the text subset of "
              "the benchmark) with it, and use a\n   different model for the "
              "visual arms.")


if __name__ == "__main__":
    main()
