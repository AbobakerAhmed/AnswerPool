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
#   python -m AnswerPooling.run_matching --in groups_n5.jsonl --model gemini-3.1-flash-lite --limit 200
#   python -m AnswerPooling.run_matching --in groups_n5_co.jsonl --model ... --limit 200   # choices-only
#   ... --groups-from h_match.gemini-3.6-flash.results.jsonl   # pin to another run's groups
#   ... --thinking-level low --max-out 16384                  # Gemini Pro, thinking cannot be off
# Rerunning a finished file calls only the records that failed with an API
# error. API answers are appended to <results>.part as they arrive.
# ============================================================================
import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# vLLM's engine-core child re-imports this module under spawn; setting this
# before any CUDA touch avoids "Cannot re-initialize CUDA in forked subprocess".
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def label(i):
    """Candidate label i: A..Z, then AA, AB, ... like spreadsheet columns.
    The first 26 are the bare letters, so every build and result produced
    before pools could exceed 26 is byte-identical. Pools above 26 arise when
    10-option items are pooled with all their distractors (N=3 gives M=30),
    which is required for matching to be a strict superset of the MCQ item."""
    if i < 26:
        return LETTERS[i]
    i -= 26
    return LETTERS[i // 26] + LETTERS[i % 26]


def labels(m):
    return [label(i) for i in range(m)]


def label_index(s):
    """Inverse of label(); -1 if s is not a label."""
    s = (s or "").strip().upper()
    if len(s) == 1 and s in LETTERS:
        return LETTERS.index(s)
    if len(s) == 2 and s[0] in LETTERS and s[1] in LETTERS:
        return 26 + LETTERS.index(s[0]) * 26 + LETTERS.index(s[1])
    return -1


def label_pat(m):
    """Regex atom for one label in a pool of size m. Stays the single-letter
    class the parsers were validated with whenever m <= 26."""
    return r"[A-Z]{1,2}" if m > 26 else r"[A-Z]"


def phash(prompt):
    """Fingerprint of the exact prompt, stored with every result so resume can
    tell whether a cached answer belongs to the input in front of it. Question
    text alone is not enough -- the same question with reshuffled options is a
    different item and the cached letter would be meaningless."""
    return hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ client --
def make_client(mode=None):
    """Three ways in, chosen by --api or RUN_API_MODE (default auto):
      vertex      service-account json named by VERTEX_KEY
      vertex-key  Vertex AI express mode, an API key enabled for Vertex AI
      studio      Gemini API (generativelanguage.googleapis.com) with an API key
    auto takes the json when VERTEX_KEY names one, else the Gemini API. The key
    prefix does not tell the two apart: a Gemini API key starting with "AQ."
    was refused by Vertex AI with 403 PERMISSION_DENIED. Express mode must be
    asked for with --api vertex-key.
    The key itself is only ever read from GOOGLE_API_KEY or GEMINI_API_KEY."""
    try:
        from google import genai
    except ImportError:
        raise SystemExit("pip install google-genai")
    mode = mode or os.environ.get("RUN_API_MODE", "auto")
    cred = os.environ.get("VERTEX_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if mode == "auto":
        if cred and os.path.exists(cred):
            mode = "vertex"
        elif key:
            mode = "studio"
        else:
            raise SystemExit("set VERTEX_KEY=/path/key.json or GOOGLE_API_KEY")
    if mode == "vertex":
        if not (cred and os.path.exists(cred)):
            raise SystemExit("--api vertex needs VERTEX_KEY=/path/key.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(cred)
        proj = os.environ.get("VERTEX_PROJECT") or json.load(open(cred)).get("project_id")
        loc = os.environ.get("VERTEX_LOCATION", "global")
        print(f"[client] Vertex | project={proj} | location={loc}")
        return genai.Client(vertexai=True, project=proj, location=loc)
    if not key:
        raise SystemExit(f"--api {mode} needs GOOGLE_API_KEY")
    if mode == "vertex-key":
        print("[client] Vertex AI express mode | API key")
        return genai.Client(vertexai=True, api_key=key)
    print("[client] Gemini API | API key")
    return genai.Client(api_key=key)


_MODE = [None]
_LEVEL = [None]      # --thinking-level, for models that cannot switch thinking off
_FIRST_ERR = [True]  # print the first API error message once, so a bad key or model name shows


def make_config(max_out):
    """Thinking must not starve the answer: a budget-starved reasoning model
    returns finish_reason=MAX_TOKENS with EMPTY text, which looks identical to
    a wrong answer. Disable thinking where the model allows it. A model that
    cannot switch it off (Gemini Pro) runs at the level --thinking-level
    names, the same treatment gpt-oss gets with reasoning effort low."""
    from google.genai import types
    kw = dict(temperature=0.0, max_output_tokens=max_out)
    if _LEVEL[0]:
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=_LEVEL[0])
        return types.GenerateContentConfig(**kw)
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


def is_transient(e):
    """Quota and overload errors clear with time. They get a longer backoff so a
    busy shared quota does not turn into error records."""
    s = str(e).lower()
    return any(t in s for t in ("quota", "resource_exhausted", "429", "503", "unavailable",
                                "overloaded", "deadline_exceeded", "internal error"))


def is_auth(e):
    """A refused key or a blocked API fails every call the same way: no retry,
    and three in a row stop the run (see _ABORT)."""
    s = str(e).lower()
    return any(t in s for t in ("permission_denied", "unauthenticated", "api_key_invalid",
                                "api key not valid", "api key expired"))


_AUTH_FAILS = [0]
_ABORT = threading.Event()


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


def load_images(rec, max_pixels=0):
    """Open the image files a record cites. Returns [] for text-only records,
    which is every record on every text benchmark, so this costs nothing there.
    A missing file is reported and skipped rather than killing the run: the
    item then scores as a text-only question, which the summary flags.
    max_pixels > 0 shrinks each image to at most that many pixels, keeping the
    aspect ratio: the API path uses it to give a model the same pixel budget
    the local runs had (--max-pixels) and to keep multi-image requests small."""
    paths = rec.get("images") or []
    if not paths:
        return []
    from PIL import Image
    resample = getattr(Image, "Resampling", Image).LANCZOS
    out = []
    for p in paths:
        try:
            im = Image.open(p)
            im.load()
            im = im.convert("RGB")
            if max_pixels and im.width * im.height > max_pixels:
                s = (max_pixels / (im.width * im.height)) ** 0.5
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), resample)
            out.append(im)
        except Exception as e:
            print(f"    [img] cannot open {p}: {type(e).__name__}")
    return out


def call(client, model, prompt, max_out, rpm, images=()):
    tries = 0
    while True:
        try:
            throttle(rpm)
            # google-genai accepts PIL images inline: the prompt cites them as
            # "<image k>" and they are appended in that same global order.
            r = client.models.generate_content(model=model,
                                               contents=[prompt, *images],
                                               config=make_config(max_out))
            return (r.text or "").strip(), None
        except Exception as e:
            # a model that refuses thinking_budget=0 runs without it. Not when
            # --thinking-level is set: a refused level must fail loudly rather
            # than fall back to the model's default, which may be high.
            if cfg_rejected(e) and _MODE[0] is None and not _LEVEL[0]:
                _MODE[0] = 1
                print("    [cfg] model rejected thinking_config -> disabled")
                continue
            if is_auth(e):
                with _lock:
                    _AUTH_FAILS[0] += 1
                    first = _AUTH_FAILS[0] == 1
                    if _AUTH_FAILS[0] >= 3:
                        _ABORT.set()
                if first:
                    print(f"    [api] {type(e).__name__}: {str(e)[:300]}")
                return "", "error:auth"
            tries += 1
            transient = is_transient(e)
            if tries >= (8 if transient else 5):
                if _FIRST_ERR[0]:
                    _FIRST_ERR[0] = False
                    print(f"    [api] {type(e).__name__}: {str(e)[:300]}")
                return "", f"error:{type(e).__name__}"
            time.sleep(min(2 ** tries, 60 if transient else 20) * (0.5 + random.random()))


# ------------------------------------------------------------------ parsing --
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
# gpt-oss (harmony format): the model writes an analysis channel and then a
# final channel. Only the final channel is the answer. Generated with
# skip_special_tokens=False so the channel markers survive.
_HARMONY_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)", re.S)


def harmony_final(text):
    """The final-channel text of a harmony response, or '' when the budget ran
    out inside the analysis channel, which is then counted as unparsed."""
    m = _HARMONY_FINAL.search(text or "")
    return m.group(1).strip() if m else ""


def strip_think(text):
    """Remove reasoning blocks before parsing. An unterminated <think> (budget
    exhausted mid-reasoning) is stripped to the end, leaving an empty string —
    which is correctly counted as unparsed rather than scored as a wrong answer."""
    if "<|channel|>" in (text or ""):
        text = harmony_final(text)
    return _THINK.sub(" ", text or "").strip()


def parse_assignment(text, n, m):
    text = strip_think(text)
    """-> list of length n with letters / 'none' / None(unparsed). Accepts
    '1: B', '1) B', '1 - none', 'Q1: B', bare lines, in any order."""
    out = [None] * n
    valid = set(labels(m))
    L = label_pat(m)          # "[A-Z]" for every pool the parsers were tuned on
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
            rf"^[\(\[]?\s*({L})\s*[\)\].,:;]?\s*$",       # the whole value is a letter
            rf"[:\-–—=]\s*[\(\[]?({L})[\)\]]?\s*$",       # "... ? : M"
            rf"\s[\(\[]?({L})[\)\]]?\s*$",                # "... ? B"
            rf"\b(?:OPTION|ANSWER|CANDIDATE)\s*[:\-]?\s*({L})\b",
            rf"[?:.…!]\s*[\(\[]?({L})[\)\].]\s",          # "... ? A. To begin ..."
            rf"^[\(\[]?\s*({L})\s*[\).:,\]]",             # "B. text" -- punctuation
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
        for mt in re.finditer(rf"\b(\d+)\s*[-:=)>]+\s*({L})\b(?![\w'])", text.upper()):
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
    m = max(1, m)
    if m <= 26:
        lab = f"[{LETTERS[:m]}]"                # the grammar every run so far used
    else:
        # two-letter labels first so "AA" is not consumed as "A" + junk
        lab = "(?:" + "|".join(sorted(labels(m), key=len, reverse=True)) + ")"
    alt = lab
    # 'none' is a legal token on matching withhold arms. NOT on mcq_none, where
    # abstention is a lettered option like any other -- allowing the bare word
    # there would give the model a second, ungraded way to abstain. On
    # mcq_prose 'none' must be legal on EVERY item, answerable ones included:
    # gating it on withheld would make false abstention physically impossible
    # and silently break the arm's error trade-off.
    arm = str(rec.get("arm", ""))
    if (rec.get("withheld") and not arm.startswith("mcq")) or arm == "mcq_prose":
        alt = f"(?:{lab}|none)"
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


def run_vllm(recs, model, max_out, tp=1, max_len=32768, gpu_util=0.90, guided=True,
             max_pixels=0, mm_kwargs=None):
    """Local open-weight models (Qwen3, Llama, Mistral, ...). Batches every
    prompt in one call — far faster than the API path, and no quota.
    Built lazily inside this function: constructing the engine at import time
    breaks under spawn, because the child re-imports the module."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # multimodal records carry image paths; a vision model needs its processor
    # to place the image placeholder tokens its template expects
    n_img = max(len(r.get("images") or []) for r in recs) if recs else 0
    proc = None
    if n_img:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        print(f"[vllm] multimodal input: up to {n_img} images per prompt")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # gpt-oss cannot switch reasoning off. It is run at the lowest reasoning
    # effort, without the grammar (which would fight the analysis channel),
    # with room for the analysis, and only its final channel is parsed.
    harmony = "gpt-oss" in model.lower()
    if harmony:
        guided = False
        max_out = max(max_out, 4096)
        print("[vllm] gpt-oss: harmony format, reasoning effort low, guided "
              "decoding OFF, final channel parsed, max_out "
              f"{max_out}")
    tpl_kw = {"reasoning_effort": "low"} if harmony else {"enable_thinking": False}
    mm_kw = {}
    if n_img:
        mm_kw["limit_mm_per_prompt"] = {"image": n_img}
        # Qwen-VL tokenises images at native resolution, so a pooled prompt
        # of 3 to 4 large MMMU-Pro screenshots can run to many thousands of
        # tokens. --max-pixels caps each image's token cost. Gemma 3 uses a
        # fixed 256 tokens per image and ignores the setting.
        if max_pixels:
            mm_kw["mm_processor_kwargs"] = {"max_pixels": max_pixels}
        # newer Qwen processors (Qwen3.5) take the pixel budget as
        # size={"longest_edge": N}; --mm-kwargs passes any processor kwargs
        if mm_kwargs:
            mm_kw["mm_processor_kwargs"] = {**mm_kw.get("mm_processor_kwargs", {}), **mm_kwargs}
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_len,
              gpu_memory_utilization=gpu_util, tensor_parallel_size=tp,
              trust_remote_code=True, **mm_kw)

    prompts, keep = [], []
    for i, rec in enumerate(recs):
        imgs = load_images(rec)
        if imgs:
            # image parts first, in the same global order the prompt cites
            content = ([{"type": "image"}] * len(imgs)
                       + [{"type": "text", "text": rec["prompt"]}])
            msg = [{"role": "user", "content": content}]
            try:
                p = (proc or tok).apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True, **tpl_kw)
            except TypeError:
                p = (proc or tok).apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True)
            # text side only: image token counts are model specific and not
            # countable here, so this under-counts. Budget accordingly.
            if len(tok.encode(p)) + max_out > max_len - 64:
                continue
            prompts.append({"prompt": p,
                            "multi_modal_data": {"image": imgs}})
            keep.append(i)
            continue
        msg = [{"role": "user", "content": rec["prompt"]}]
        # Qwen3 (and others) enable THINKING by default: the model emits
        # <think>...</think> before the answer, burns the token budget, and
        # single-letter parsing lands inside the reasoning. Turn it off where
        # the template supports it.
        try:
            p = tok.apply_chat_template(msg, tokenize=False,
                                        add_generation_prompt=True, **tpl_kw)
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
        sp = SamplingParams(temperature=0.0, max_tokens=max_out,
                            skip_special_tokens=not harmony)
    outs = llm.generate(prompts, sp)
    texts = {i: (harmony_final(o.outputs[0].text) if harmony
                 else o.outputs[0].text.strip())
             for i, o in zip(keep, outs)}
    if harmony:
        empty = sum(1 for t in texts.values() if not t)
        print(f"[vllm] gpt-oss: {empty}/{len(texts)} responses had no final "
              f"channel (analysis ran past max_out), counted as unparsed")
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
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="vllm, multimodal only: cap each image's pixel count "
                         "before tokenisation (Qwen-VL honours it, Gemma 3 "
                         "ignores it). 1003520 = 1280x784, a safe default for "
                         "pooled prompts of 3 to 4 images at --max-len 32768")
    ap.add_argument("--mm-kwargs", default="",
                    help="JSON of extra multimodal processor kwargs, for "
                         "processors that take the pixel budget as "
                         "size={longest_edge: N} instead of max_pixels (Qwen3.5)")
    ap.add_argument("--no-guided", action="store_true",
                    help="vllm: do NOT constrain output to the answer grammar "
                         "(measures format compliance as well as capability)")
    ap.add_argument("--groups-from", default="",
                    help="run only the groups whose ids appear in this jsonl (a build or a "
                         "results file). Pins a run to the groups another model ran, e.g. the "
                         "300 QuALITY groups Gemini ran with --limit 300. --limit cannot do "
                         "this on an MCQ file, where the first records are questions, not groups")
    ap.add_argument("--thinking-level", default="", choices=["", "minimal", "low", "medium", "high"],
                    help="api only: for a model that cannot switch thinking off (Gemini Pro), "
                         "run it at this level instead of its default. Use a large --max-out, "
                         "thinking tokens count against it")
    ap.add_argument("--api", default="auto", choices=["auto", "vertex", "vertex-key", "studio"],
                    help="api backend only: how to reach Google, see make_client")
    ap.add_argument("--name", default="",
                    help="model label for the results file name, default --model. Keeps one "
                         "file per model when an endpoint names it differently, e.g. "
                         "--model gemma-4-26b-a4b-it --name gemma-4-26b-a4b-it-maas")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.api != "auto":
        os.environ["RUN_API_MODE"] = args.api

    if args.thinking_level:
        if args.backend != "api":
            raise SystemExit("--thinking-level applies to the api backend only")
        try:
            from google.genai import types
            types.ThinkingConfig(thinking_level=args.thinking_level.upper())
        except Exception as e:
            raise SystemExit("--thinking-level needs a google-genai that knows thinking_level "
                             f"(pip install -U google-genai): {e}")
        _LEVEL[0] = args.thinking_level.upper()

    recs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    if args.groups_from:
        gids = {json.loads(l)["group_id"] for l in open(args.groups_from, encoding="utf-8")
                if l.strip()}
        before = len(recs)
        recs = [r for r in recs if r["group_id"] in gids]
        print(f"--groups-from: {len(recs)} of {before} records, the {len(gids)} groups "
              f"of {args.groups_from}")
        if not recs:
            raise SystemExit("no record of the input belongs to those groups")
    if args.limit:
        recs = recs[:args.limit]
    arm = recs[0]["arm"]
    n, m = recs[0]["n"], recs[0]["m"]
    out_path = args.out or args.inp.replace(
        ".jsonl", f".{(args.name or args.model).replace('/','_')}.results.jsonl")
    # API runs append every answer here as it arrives, so a crash or a Ctrl-C
    # loses nothing: the next run resumes from it. Removed after the final
    # write. The collectors glob *.results.jsonl and never read it.
    part_path = out_path + ".part"

    done = {}
    for path in (out_path, part_path):
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                try:
                    r = json.loads(l); done[r["key"]] = r
                except Exception:
                    pass
    if done:
        # A cached entry is only reusable if it answered the SAME question.
        # item_id is "{group_id}_q{i}", which a rebuilt input reproduces exactly
        # even when the questions inside a group have changed -- so key-only
        # resume silently pairs old answers with new questions and reports a
        # complete run. Verify the text before trusting the cache.
        stale = retry = 0
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
                # a record skipped for length or lost to an API error carries
                # no answer: never trust it, so a rerun (a larger --max-len, a
                # working key) redoes exactly those
                elif str(done[k].get("err") or "").startswith(("skipped", "error")):
                    del done[k]
                    retry += 1
        print(f"resuming: {len(done)} already done"
              + (f"  ({stale} DISCARDED — cached answers were for different "
                 f"questions; the input was rebuilt)" if stale else "")
              + (f"  ({retry} to retry: API error or skipped for length)" if retry else ""))

    print(f"arm={arm} N={n} M={m} | {len(recs)} records | model={args.model} "
          f"| backend={args.backend}")

    def score_one(rec, txt, err):
        key = rec.get("item_id") or rec["group_id"]
        if arm == "freeform":
            # free text -- there is nothing to parse. A separate judge assigns
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
                    none_letter=rec.get("none_letter", ""),   # printed none option
                    phash=phash(rec["prompt"]),           # resume-safety, see above
                    q_index=rec.get("q_index"),           # per-question join key
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
                         guided=not args.no_guided,
                         max_pixels=args.max_pixels,
                         mm_kwargs=(json.loads(args.mm_kwargs) if args.mm_kwargs else None)
                         ) if todo else {}
        results = [done[(r.get("item_id") or r["group_id"])]
                   for r in recs if (r.get("item_id") or r["group_id"]) in done]
        for i, rec in enumerate(todo):
            results.append(score_one(rec, texts.get(i, ""),
                                     None if i in texts else "skipped:too_long"))
    else:
        client = make_client()
        plock = threading.Lock()
        part = open(part_path, "a", encoding="utf-8")

        def work(rec):
            key = rec.get("item_id") or rec["group_id"]
            if key in done:
                return done[key]
            if _ABORT.is_set():
                return None
            txt, err = call(client, args.model, rec["prompt"], args.max_out,
                            args.rpm, images=load_images(rec, args.max_pixels))
            if err == "error:auth":
                return None
            res = score_one(rec, txt, err)
            with plock:
                if not part.closed:
                    part.write(json.dumps(res, ensure_ascii=False) + "\n")
                    part.flush()
            return res

        ex = ThreadPoolExecutor(max_workers=args.workers)
        try:
            results = list(ex.map(work, recs))
            ex.shutdown()
        except BaseException:
            # Ctrl-C: drop the queued calls instead of finishing them all first
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            with plock:
                part.close()
        if _ABORT.is_set() or any(r is None for r in results):
            raise SystemExit(
                "stopped: the API refused the key or the endpoint (the [api] line above). "
                f"Answers received so far are kept in {part_path}. Fix --api or the key and "
                "rerun, only the missing records are called.")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if os.path.exists(part_path):
        os.remove(part_path)

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
        print("NOT SCORED YET: free-text responses are scored by a separate judge")
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
