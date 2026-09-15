#!/usr/bin/env python
# ============================================================================
# run_matching.py — run a built .jsonl arm against a model and score it.
#
# Metrics (see MATCHING_EVAL.md):
#   per-pair accuracy      floor 1/M      (partial credit, statistical power)
#   exact-assignment rate  floor 1/(M!/(M-N)!)   (headline)
#   false-match rate       assigned an answer to a WITHHELD question -> confabulation
#   miss rate              said "none" when the gold WAS present
#   parse-failure rate     reported SEPARATELY from wrong answers
#
# Client: VERTEX_KEY=/path/key.json  or  GOOGLE_API_KEY=...
#
#   python run_matching.py --in groups_n5.jsonl --model gemini-3.1-flash-lite --limit 200
#   python run_matching.py --in groups_n5_co.jsonl --model ... --limit 200   # choices-only
# ============================================================================
import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# vLLM's engine-core child re-imports this module under spawn; setting this
# before any CUDA touch avoids "Cannot re-initialize CUDA in forked subprocess".
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def phash(prompt):
    """Fingerprint of the exact prompt, stored with every result so resume can
    tell whether a cached answer belongs to the input in front of it. Question
    text alone is not enough -- the same question with reshuffled options is a
    different item and the cached letter would be meaningless."""
    return hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ client --
def make_client():
    try:
        from google import genai
    except ImportError:
        raise SystemExit("pip install google-genai")
    cred = os.environ.get("VERTEX_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred and os.path.exists(cred):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(cred)
        proj = os.environ.get("VERTEX_PROJECT") or json.load(open(cred)).get("project_id")
        loc = os.environ.get("VERTEX_LOCATION", "global")
        print(f"[client] Vertex | project={proj} | location={loc}")
        return genai.Client(vertexai=True, project=proj, location=loc)
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set VERTEX_KEY=/path/key.json or GOOGLE_API_KEY")
    print("[client] Gemini API key")
    return genai.Client(api_key=key)


_MODE = [None]


def make_config(max_out):
    """Thinking must not starve the answer: a budget-starved reasoning model
    returns finish_reason=MAX_TOKENS with EMPTY text, which looks identical to
    a wrong answer. Disable thinking where the model allows it."""
    from google.genai import types
    kw = dict(temperature=0.0, max_output_tokens=max_out)
    if _MODE[0] in (None, 0):
        try:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            return types.GenerateContentConfig(**kw)
        except Exception:
            pass
    return types.GenerateContentConfig(**kw)


def cfg_rejected(e):
    s = str(e).lower()
    if any(t in s for t in ("quota", "resource_exhausted", "429")):
        return False
    return any(t in s for t in ("thinking", "not supported", "unknown field", "invalid_argument"))


# --------------------------------------------------------------- inference --
_next, _lock = [0.0], threading.Lock()


def throttle(rpm):
    if rpm <= 0:
        return
    gap = 60.0 / rpm
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + gap
    if wait:
        time.sleep(wait)


def call(client, model, prompt, max_out, rpm):
    for attempt in range(5):
        try:
            throttle(rpm)
            r = client.models.generate_content(model=model, contents=[prompt],
                                               config=make_config(max_out))
            return (r.text or "").strip(), None
        except Exception as e:
            if cfg_rejected(e) and _MODE[0] is None:
                _MODE[0] = 1
                print("    [cfg] model rejected thinking_config -> disabled")
                continue
            if attempt == 4:
                return "", f"error:{type(e).__name__}"
            time.sleep(min(2 ** attempt, 20))
    return "", "error:retries"


# ------------------------------------------------------------------ parsing --
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)


def strip_think(text):
    """Remove reasoning blocks before parsing. An unterminated <think> (budget
    exhausted mid-reasoning) is stripped to the end, leaving an empty string —
    which is correctly counted as unparsed rather than scored as a wrong answer."""
    return _THINK.sub(" ", text or "").strip()


def parse_assignment(text, n, m):
    text = strip_think(text)
    """-> list of length n with letters / 'none' / None(unparsed). Accepts
    '1: B', '1) B', '1 - none', 'Q1: B', bare lines, in any order."""
    out = [None] * n
    valid = set(LETTERS[:m])
    # strip markdown emphasis/code marks: `1: **B**` was failing the bare-letter
    # match, which is most of Qwen3's "unparsed" slots.
    text = re.sub(r"[*_`\"'#]+", "", text)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # separators seen in the wild: 1: B / 1) B / 1. B / 1 - B / 1 -> B / 1 = B
        mt = re.match(r"^\s*(?:Q(?:uestion)?\s*)?(\d+)\s*(?:[):.\-=]|->|→|:)\s*(.+)$",
                      line, re.I)
        if not mt:
            continue
        idx = int(mt.group(1)) - 1
        if not (0 <= idx < n):
            continue
        val = mt.group(2).strip()
        if re.match(r"^none\b|^n/a\b|^-+$", val, re.I):
            out[idx] = "none"
            continue
        u = val.upper()
        # Qwen3 ECHOES the question before answering, so the letter is at the END
        # of the line, not the start:
        #   "1. How did the Ruler become the Ruler? B"
        #   "1. ...under lie detector questioning? : M"
        #   "1. Why did Duane ring the bell? A. To begin his escape plan"
        # Reading group(2)'s first token gets question text every time. These are
        # ordered most-specific first; a plain leading letter is checked LAST so a
        # question starting with "A ..." can't win over a real trailing marker.
        pats = (
            r"^[\(\[]?\s*([A-Z])\s*[\)\].,:;]?\s*$",       # the whole value is a letter
            r"[:\-–—=]\s*[\(\[]?([A-Z])[\)\]]?\s*$",       # "... ? : M"
            r"\s[\(\[]?([A-Z])[\)\]]?\s*$",                # "... ? B"
            r"\b(?:OPTION|ANSWER|CANDIDATE)\s*[:\-]?\s*([A-Z])\b",
            r"[?:.…!]\s*[\(\[]?([A-Z])[\)\].]\s",          # "... ? A. To begin ..."
            r"^[\(\[]?\s*([A-Z])\s*[\).:,\]]",             # "B. text" -- punctuation
        )                                                  # after the letter is
        #  REQUIRED: "1. A ship arrives at dawn" is an echoed question with no
        #  answer, and a bare `^([A-Z])\b` turns it into a confident "A".
        for p in pats:
            lm = re.search(p, u)
            if lm and lm.group(1) in valid:
                out[idx] = lm.group(1); break

    # Last resort for slots still unfilled: some runs list every question first
    # and collect the answers in a trailing block ("1-B, 2-D, 3-C"). Scan the
    # whole text for numbered pairs. '.' is NOT an accepted separator here --
    # "1. A big house" would parse as A -- so this can only fire on markers.
    if any(p is None for p in out):
        for mt in re.finditer(r"\b(\d+)\s*[-:=)>]+\s*([A-Z])\b(?![\w'])", text.upper()):
            idx = int(mt.group(1)) - 1
            if 0 <= idx < n and out[idx] is None and mt.group(2) in valid:
                out[idx] = mt.group(2)
    return out


def parse_single(text, m):
    """Read the answer from the END of the output, not the start. Searching
    forward finds the first standalone capital anywhere -- inside reasoning,
    a bulleted list, or the word 'I' -- which is how Qwen3 scored 0/1500."""
    valid = set(LETTERS[:m])
    text = strip_think(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines[-3:]):                    # answer is usually last
        u = line.upper()
        mt = re.fullmatch(r"[\(\[\*\s]*([A-Z])[\)\].,:\*\s]*", u)
        if mt and mt.group(1) in valid:
            return mt.group(1)
        mt = re.search(r"(?:ANSWER|OPTION)\D{0,4}\b([A-Z])\b", u)
        if mt and mt.group(1) in valid:
            return mt.group(1)
    for u in (l.upper() for l in reversed(lines)):       # last standalone letter
        for mt in reversed(list(re.finditer(r"\b([A-Z])\b", u))):
            if mt.group(1) in valid:
                return mt.group(1)
    return None


# ------------------------------------------------------------ vLLM backend --
def answer_regex(rec):
    """Exact output grammar for one record: '1: B\\n2: D\\n...'.

    Gemini follows the requested format; Qwen3-8B does not -- it echoes each
    question, wanders into prose, or answers nothing, and no regex over free
    text can reliably tell an answer from an echoed question. Constraining the
    DECODER removes the problem at the source instead: format compliance
    becomes 100% by construction and parse rate stops being a confound in the
    accuracy numbers.
    """
    n = int(rec.get("n") or len(rec.get("answer", [])) or 1)
    m = int(rec.get("m") or len(rec.get("candidates", [])) or 4)
    m = max(1, min(m, 26))
    letters = LETTERS[:m]
    alt = f"[{letters}]"
    # 'none' is a legal token on matching withhold arms. NOT on mcq_none, where
    # abstention is a lettered option like any other -- allowing the bare word
    # there would give the model a second, ungraded way to abstain. On
    # mcq_prose 'none' must be legal on EVERY item, answerable ones included:
    # gating it on withheld would make false abstention physically impossible
    # and silently break the arm's error trade-off.
    arm = str(rec.get("arm", ""))
    if (rec.get("withheld") and not arm.startswith("mcq")) or arm == "mcq_prose":
        alt = f"(?:[{letters}]|none)"
    if n <= 1:
        return alt
    return r"\n".join(f"{i+1}: {alt}" for i in range(n))


def _guided_kwargs(regex):
    """vLLM renamed this between versions; try both, degrade to unconstrained."""
    try:
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(regex=regex)}
    except Exception:
        pass
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(regex=regex)}
    except Exception:
        return None


def run_vllm(recs, model, max_out, tp=1, max_len=32768, gpu_util=0.90, guided=True):
    """Local open-weight models (Qwen3, Llama, Mistral, ...). Batches every
    prompt in one call — far faster than the API path, and no quota.
    Built lazily inside this function: constructing the engine at import time
    breaks under spawn, because the child re-imports the module."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_len,
              gpu_memory_utilization=gpu_util, tensor_parallel_size=tp,
              trust_remote_code=True)

    prompts, keep = [], []
    for i, rec in enumerate(recs):
        msg = [{"role": "user", "content": rec["prompt"]}]
        # Qwen3 (and others) enable THINKING by default: the model emits
        # <think>...</think> before the answer, burns the token budget, and
        # single-letter parsing lands inside the reasoning. Turn it off where
        # the template supports it.
        try:
            p = tok.apply_chat_template(msg, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            try:
                p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            except Exception:                   # base models without a template
                p = rec["prompt"]
        except Exception:
            p = rec["prompt"]
        if len(tok.encode(p)) + max_out > max_len - 64:
            continue                            # would be rejected; leave unanswered
        prompts.append(p); keep.append(i)

    if guided and recs and recs[0].get("arm") == "freeform":
        guided = False                     # free text has no grammar to enforce
        print("[vllm] guided decoding OFF for freeform (open-ended by design)")
    if guided:
        sps, ok = [], True
        for i in keep:
            kw = _guided_kwargs(answer_regex(recs[i]))
            if kw is None:
                ok = False
                break
            sps.append(SamplingParams(temperature=0.0, max_tokens=max_out, **kw))
        if ok:
            sp = sps
            print(f"[vllm] guided decoding ON — output format is enforced, "
                  f"so unparsed slots should be 0")
        else:
            sp = SamplingParams(temperature=0.0, max_tokens=max_out)
            print("[vllm] WARNING: this vLLM build exposes no guided-decoding API; "
                  "falling back to free text. Parse rate is a confound again.")
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=max_out)
    outs = llm.generate(prompts, sp)
    texts = {i: o.outputs[0].text.strip() for i, o in zip(keep, outs)}
    print(f"[vllm] generated {len(texts)}/{len(recs)} "
          f"({len(recs)-len(texts)} skipped: prompt too long)")
    return texts


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=0)
    ap.add_argument("--max-out", type=int, default=2048)
    ap.add_argument("--backend", default="api", choices=["api", "vllm"],
                    help="vllm = local open-weight model (Qwen3, Llama, ...)")
    ap.add_argument("--tp", type=int, default=1, help="vllm tensor-parallel size")
    ap.add_argument("--max-len", type=int, default=32768, help="vllm max_model_len")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--no-guided", action="store_true",
                    help="vllm: do NOT constrain output to the answer grammar "
                         "(measures format compliance as well as capability)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    arm = recs[0]["arm"]
    n, m = recs[0]["n"], recs[0]["m"]
    out_path = args.out or args.inp.replace(".jsonl", f".{args.model.replace('/','_')}.results.jsonl")

    done = {}
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(l); done[r["key"]] = r
            except Exception:
                pass
        # A cached entry is only reusable if it answered the SAME question.
        # item_id is "{group_id}_q{i}", which a rebuilt input reproduces exactly
        # even when the questions inside a group have changed -- so key-only
        # resume silently pairs old answers with new questions and reports a
        # complete run. Verify the text before trusting the cache.
        stale = 0
        for rec in recs:
            k = rec.get("item_id") or rec["group_id"]
            if k in done:
                cached, now = done[k].get("phash"), phash(rec["prompt"])
                # fall back to question text for results written before phash
                bad = (cached != now) if cached else (
                    done[k].get("questions") != rec.get("questions"))
                if bad:
                    del done[k]
                    stale += 1
        print(f"resuming: {len(done)} already done"
              + (f"  ({stale} DISCARDED — cached answers were for different "
                 f"questions; the input was rebuilt)" if stale else ""))

    print(f"arm={arm} N={n} M={m} | {len(recs)} records | model={args.model} "
          f"| backend={args.backend}")

    def score_one(rec, txt, err):
        key = rec.get("item_id") or rec["group_id"]
        if arm == "freeform":
            # free text -- there is nothing to parse. judge_freeform.py assigns
            # ABSTAIN/ANSWER afterwards; leaving pred None here keeps unjudged
            # records from being silently scored as wrong.
            pred = [None]
        elif arm.startswith("mcq"):        # mcq arms emit a bare letter
            body = strip_think(txt or "").strip()
            # mcq_prose abstains with the literal word, not a letter
            if arm == "mcq_prose" and re.fullmatch(r"[\"'\s]*none[\"'.\s]*", body, re.I):
                pred = ["none"]
            else:
                pred = [parse_single(txt, rec["m"])]
                if arm == "mcq_prose" and pred[0] is None \
                        and re.search(r"\bnone\b", body, re.I):
                    pred = ["none"]
            # mcq_none offers "none of these" as a lettered option in a shuffled
            # position. Normalise that letter to the string "none" so abstention
            # is scored by the same false-match/miss code as the matching arms --
            # otherwise the two formats are compared on different conventions.
            nl = rec.get("none_letter")
            if nl and pred[0] == nl:
                pred = ["none"]
        else:
            pred = parse_assignment(txt, rec["n"], rec["m"])
        return dict(key=key, group_id=rec["group_id"], arm=arm,
                    withheld=rec.get("withheld", []), gold=rec["answer"],
                    questions=rec.get("questions", []),   # needed by filter_blind.py
                    n=rec.get("n"), m=rec.get("m"),       # pool size -> rescore.py
                    none_letter=rec.get("none_letter", ""),   # -> position_bias.py
                    phash=phash(rec["prompt"]),           # resume-safety, see above
                    q_index=rec.get("q_index"),           # join key for validity.py
                    # raw was capped at 300 chars, which silently DECAPITATED any
                    # model that echoes the questions before answering (Qwen3 does;
                    # Gemini's "1: B\n2: D" never hit the cap). The saved text then
                    # lost the last answers and no offline rescore could recover
                    # them. Keep enough to hold a full response.
                    pred=pred, err=err, raw=txt[:4000])

    t0 = time.perf_counter()
    if args.backend == "vllm":
        todo = [r for r in recs if (r.get("item_id") or r["group_id"]) not in done]
        texts = run_vllm(todo, args.model, args.max_out, args.tp,
                         args.max_len, args.gpu_util,
                         guided=not args.no_guided) if todo else {}
        results = [done[(r.get("item_id") or r["group_id"])]
                   for r in recs if (r.get("item_id") or r["group_id"]) in done]
        for i, rec in enumerate(todo):
            results.append(score_one(rec, texts.get(i, ""),
                                     None if i in texts else "skipped:too_long"))
    else:
        client = make_client()

        def work(rec):
            key = rec.get("item_id") or rec["group_id"]
            if key in done:
                return done[key]
            txt, err = call(client, args.model, rec["prompt"], args.max_out, args.rpm)
            return score_one(rec, txt, err)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(work, recs))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------- scoring --
    pair_ok = pair_tot = 0
    exact_ok = exact_tot = 0
    fm_ok = fm_tot = 0          # withheld: correct = "none"
    miss = present_tot = 0      # present-gold questions answered "none"
    unparsed = errs = 0
    for r in results:
        if r.get("err"):
            errs += 1
        gold, pred, wh = r["gold"], r["pred"], set(r.get("withheld", []))
        all_ok = True
        for i, (g, p) in enumerate(zip(gold, pred)):
            if p is None:
                unparsed += 1; all_ok = False
                continue
            if i in wh:
                fm_tot += 1
                if p == "none":
                    fm_ok += 1
                else:
                    all_ok = False
            else:
                present_tot += 1
                if p == "none":
                    miss += 1; all_ok = False
                pair_tot += 1
                if p == g:
                    pair_ok += 1
                else:
                    all_ok = False
        exact_tot += 1
        exact_ok += all_ok

    inj = m if arm.startswith("mcq") else math.prod(range(m - n + 1, m + 1))
    print("\n" + "=" * 66)
    print(f"{arm}  N={n} M={m}  [{time.perf_counter()-t0:.0f}s]")
    print("=" * 66)
    if arm == "freeform":
        # Nothing here is scoreable yet: pred is deliberately None until the
        # judge runs, so accuracy 0/0 and "all slots unparsed" are the correct
        # state, not a failed run. Say so rather than printing zeros that read
        # like a catastrophe.
        n_un = sum(1 for r in results if r.get("withheld"))
        print(f"stored {len(results)} free-text responses "
              f"({n_un} on unanswerable items, {len(results)-n_un} answerable)")
        print(f"api errors {errs}")
        print("NOT SCORED YET -> python judge_freeform.py "
              f"--glob \"{os.path.basename(out_path)}\"")
    else:
        print(f"per-pair accuracy  {pair_ok}/{pair_tot} = {pair_ok/max(1,pair_tot):.4f}"
              f"   (chance {1/m:.4f})")
        print(f"exact assignment   {exact_ok}/{exact_tot} = "
              f"{exact_ok/max(1,exact_tot):.4f}   (chance {1/inj:.2e})")
        if fm_tot:
            print(f"FALSE-MATCH rate   {fm_tot-fm_ok}/{fm_tot} = "
                  f"{(fm_tot-fm_ok)/fm_tot:.4f}"
                  f"   <- assigned an answer to a question with NO answer in pool")
        if present_tot:
            print(f"miss rate          {miss}/{present_tot} = {miss/present_tot:.4f}"
                  f"   <- said 'none' when the gold WAS present")
        print(f"unparsed slots     {unparsed}   |  api errors {errs}"
              f"    (reported separately)")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
