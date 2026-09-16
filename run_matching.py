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
# Client: VERTEX_KEY=/path/key.json  or  GOOGLE_API_KEY=...   (--backend api)
#         OPENAI_API_KEY=... [OPENAI_BASE_URL=...]             (--backend openai)
#           OpenAI GPT-5.x, Kimi K3 (base_url https://api.moonshot.ai/v1),
#           OpenRouter, a local `vllm serve`, ...
#         ANTHROPIC_API_KEY=...                                 (--backend anthropic)
#         local weights                                         (--backend vllm)
# Visual benchmarks (MMMU, MMMU-Pro, MathVista, ScienceQA): the record's
# 'images' paths are attached to the request on every backend.
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


def phash(prompt, images=None):
    """Fingerprint of the exact prompt, stored with every result so resume can
    tell whether a cached answer belongs to the input in front of it. Question
    text alone is not enough -- the same question with reshuffled options is a
    different item and the cached letter would be meaningless. Image paths are
    part of the fingerprint on visual benchmarks for the same reason."""
    h = hashlib.sha1((prompt or "").encode("utf-8"))
    if images and any(images):
        h.update(json.dumps(images).encode("utf-8"))
    return h.hexdigest()[:16]


# ------------------------------------------------------------- multimodal --
def image_parts(rec):
    """-> [(label, PIL.Image)] for a visual-benchmark record. rec['images']
    is one list of file paths per question, labelled by question number.
    Text-only records return []. Unreadable files are skipped with a
    warning rather than raised, so one bad image doesn't abort a whole
    batched run (matters most for paid API calls in verify_wellformed.py)."""
    imgs = rec.get("images") or []
    if not any(imgs):
        return []
    from PIL import Image
    n = len(imgs)
    parts = []
    for qi, paths in enumerate(imgs):
        for k, path in enumerate(paths, 1):
            if n > 1:
                label = f"Question {qi+1}, image {k}:"
            else:
                label = f"Image {k}:" if len(paths) > 1 else "Image:"
            try:
                im = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"[image_parts] SKIPPING unreadable image {path} "
                      f"({type(e).__name__}: {e})")
                continue
            parts.append((label, im))
    return parts


def image_data_url(im):
    import base64, io
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def openai_content(rec):
    """OpenAI-style content list: labelled images first, then the prompt."""
    parts = image_parts(rec)
    if not parts:
        return rec["prompt"]
    content = []
    for label, im in parts:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(im)}})
    content.append({"type": "text", "text": rec["prompt"]})
    return content


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


# Provider errors carry the reason (bad key, no credit, unsupported image
# format, rate limit). Storing only the exception class name threw that away
# and left a run of 200 failures with nothing to debug. Keep the message, and
# print the FIRST one immediately so a doomed run is visible in seconds
# instead of after the retry backoff has burned 25 minutes.
_ERR_SHOWN = []


def note_error(e, where=""):
    msg = " ".join(str(e).split())[:400]
    if not _ERR_SHOWN:
        _ERR_SHOWN.append(msg)
        print(f"\n    [!] first API error{' from ' + where if where else ''}: "
              f"{type(e).__name__}: {msg}\n", flush=True)
    return f"error:{type(e).__name__}:{msg[:200]}"


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


def call(client, model, rec, max_out, rpm):
    # google-genai accepts a mixed list of str and PIL images as contents
    contents = []
    for label, im in image_parts(rec):
        contents += [label, im]
    contents.append(rec["prompt"])
    for attempt in range(5):
        try:
            throttle(rpm)
            r = client.models.generate_content(model=model, contents=contents,
                                               config=make_config(max_out))
            return (r.text or "").strip(), None
        except Exception as e:
            if cfg_rejected(e) and _MODE[0] is None:
                _MODE[0] = 1
                print("    [cfg] model rejected thinking_config -> disabled")
                continue
            if attempt == 4:
                return "", note_error(e, "gemini")
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


def anthropic_content(rec):
    """Anthropic content blocks: labelled images first, then the prompt."""
    parts = image_parts(rec)
    if not parts:
        return rec["prompt"]
    import base64, io
    blocks = []
    for label, im in parts:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        blocks.append({"type": "text", "text": label})
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(buf.getvalue()).decode("ascii")}})
    blocks.append({"type": "text", "text": rec["prompt"]})
    return blocks


# ------------------------------------------------------- Anthropic backend --
def make_anthropic_client():
    """Claude models (Opus 5, Sonnet 5, Fable 5.1, ...). Set ANTHROPIC_API_KEY."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("set ANTHROPIC_API_KEY")
    print("[client] Anthropic")
    return anthropic.Anthropic(api_key=key)


def call_anthropic(client, model, rec, max_out, rpm):
    """Extended thinking is left OFF: the paper's protocol disables thinking so
    that the measured quantity is the answer, not a reasoning budget, and a
    budget-starved reasoning pass returns empty text that scores as a wrong
    answer rather than as a format failure (see make_config)."""
    msgs = [{"role": "user", "content": anthropic_content(rec)}]
    for attempt in range(5):
        try:
            throttle(rpm)
            r = client.messages.create(model=model, max_tokens=max_out,
                                       temperature=0.0, messages=msgs)
            # content is a list of blocks; keep the text ones, skip thinking
            txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return txt.strip(), None
        except Exception as e:
            s_e = str(e).lower()
            if "overloaded" in s_e or "rate" in s_e or "429" in s_e:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            if attempt == 4:
                return "", note_error(e, "anthropic")
            time.sleep(min(2 ** attempt, 20))
    return "", "error:retries"


# ---------------------------------------------------------- OpenAI backend --
# Several providers speak the OpenAI protocol, and their keys are NOT
# interchangeable: a Moonshot key sent to api.openai.com is a 401 and vice
# versa. Pick the key variable that matches the endpoint, so a machine can
# hold keys for several providers at once.
OPENAI_KEY_ENVS = [
    ("moonshot", ["MOONSHOT_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY"]),
    ("kimi",     ["MOONSHOT_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY"]),
    ("openrouter", ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]),
    ("deepseek", ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]),
    ("together", ["TOGETHER_API_KEY", "OPENAI_API_KEY"]),
    # free tier: a GitHub PAT with the "Models" permission, not an OpenAI key
    ("models.github.ai", ["GITHUB_TOKEN", "GH_TOKEN"]),
]


def openai_key_for(base):
    """-> (key, env var it came from). Local servers accept any string."""
    b = (base or "").lower()
    names = ["OPENAI_API_KEY"]
    for token, envs in OPENAI_KEY_ENVS:
        if token in b:
            names = envs
            break
    for n in names:
        v = os.environ.get(n)
        if v:
            return v, n
    if base and ("localhost" in b or "127.0.0.1" in b or "0.0.0.0" in b):
        return "EMPTY", "(local server, no key needed)"
    return None, " or ".join(names)


def make_openai_client():
    """Any OpenAI-compatible endpoint: OpenAI itself, Moonshot/Kimi,
    OpenRouter, `vllm serve`, LM Studio, ... Set the provider's key variable
    and, for non-OpenAI servers, OPENAI_BASE_URL (Kimi:
    https://api.moonshot.ai/v1)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")
    base = os.environ.get("OPENAI_BASE_URL") or None
    key, src = openai_key_for(base)
    if not key:
        raise SystemExit(
            f"no API key for base_url={base or 'api.openai.com'}: set {src}")
    print(f"[client] OpenAI-compatible | base_url={base or 'api.openai.com'} "
          f"| key from {src}")
    return OpenAI(api_key=key, base_url=base)


# Reasoning models (GPT-5 family and others) reject temperature and rename the
# token cap. Both are discovered once from the first error and remembered, so
# the whole run does not pay a failed call per item.
_OAI = {"cap": "max_tokens", "temp": True}


def call_openai(client, model, rec, max_out, rpm, effort=""):
    msgs = [{"role": "user", "content": openai_content(rec)}]
    for attempt in range(5):
        try:
            throttle(rpm)
            kw = dict(model=model, messages=msgs)
            if _OAI["temp"]:
                kw["temperature"] = 0.0
            if effort:
                kw["reasoning_effort"] = effort
            kw[_OAI["cap"]] = max_out
            r = client.chat.completions.create(**kw)
            return (r.choices[0].message.content or "").strip(), None
        except Exception as e:
            msg = str(e)
            if "max_completion_tokens" in msg and _OAI["cap"] == "max_tokens":
                _OAI["cap"] = "max_completion_tokens"
                print("    [cfg] reasoning model -> max_completion_tokens")
                continue
            if "temperature" in msg and _OAI["temp"]:
                _OAI["temp"] = False
                print("    [cfg] model rejects temperature -> omitted "
                      "(decoding is no longer greedy; see --help)")
                continue
            if attempt == 4:
                return "", note_error(e, "openai-compatible")
            time.sleep(min(2 ** attempt, 20))
    return "", "error:retries"


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


def pick_dtype(requested="auto"):
    """T4 and other pre-Ampere cards (compute capability < 8.0, which is what
    Kaggle and Colab hand out for free) have no bfloat16 support: vLLM either
    refuses to start or silently upcasts. Detect and use float16 there."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name(0)
            if major < 8:
                print(f"[vllm] {name} (sm_{major}{minor}) has no bfloat16 -> dtype=float16")
                return "float16"
            print(f"[vllm] {name} (sm_{major}{minor}) -> dtype=bfloat16")
    except Exception:
        pass
    return "bfloat16"


def _strip_rope_conflict(obj, path="config"):
    """Recursively drop a config object's legacy 'type' key wherever it
    coexists with the modern 'rope_type' key (vision-language configs nest
    a sub-config, e.g. .text_config, so this isn't just top-level).
    -> number of sites patched."""
    n = 0
    if not hasattr(obj, "__dict__"):
        return n
    for k, v in list(vars(obj).items()):
        if isinstance(v, dict) and "type" in v and "rope_type" in v:
            print(f"[rope-patch] {path}.{k}: dropping legacy type={v['type']!r}, "
                  f"keeping rope_type={v['rope_type']!r}")
            del v["type"]
            n += 1
        elif hasattr(v, "__dict__"):
            n += _strip_rope_conflict(v, f"{path}.{k}")
    return n


def patch_rope_conflict(model):
    """Some vision models (Qwen2.5-VL and relatives) trigger vLLM's
    ValidationError: 'Found conflicts between rope_type=... and type=...'.
    The conflict only exists in the in-memory config object (transformers'
    AutoConfig loader adds the modern field alongside the legacy one at
    parse time), not in the raw config.json on disk, so this loads via
    AutoConfig, strips it, and saves back into the HF cache in place --
    no extra download, no separate patched directory. Returns `model`
    unchanged; the cache is what's patched."""
    import os
    try:
        from transformers import AutoConfig
    except ImportError:
        return model

    try:
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        print(f"[rope-patch] could not load config for {model} "
              f"({type(e).__name__}); leaving as-is, vLLM may still fail")
        return model

    n = _strip_rope_conflict(cfg)
    if not n:
        return model  # nothing to patch anywhere in the config tree

    if os.path.isdir(model):
        snap_dir = model
    else:
        try:
            from transformers.utils import cached_file
            snap_dir = os.path.dirname(cached_file(model, "config.json"))
        except Exception as e:
            print(f"[rope-patch] patched {n} conflict(s) in memory but could "
                  f"not locate the cache to save them ({type(e).__name__}); "
                  f"vLLM will re-derive the same conflict and still fail")
            return model

    cfg.save_pretrained(snap_dir)
    print(f"[rope-patch] patched {n} conflict(s), saved to cache at {snap_dir}")
    return model


def run_vllm(recs, model, max_out, tp=1, max_len=32768, gpu_util=0.90, guided=True,
             dtype="auto", enforce_eager=False):
    """Local open-weight models (Qwen3, Llama, Mistral, ...). Batches every
    prompt in one call. Built lazily: constructing the engine at import
    time breaks under spawn, since the child re-imports the module.

    enforce_eager: skip vLLM's CUDA-graph/torch.compile step, which needs
    scratch memory on top of the model+cache and can OOM a card that's
    otherwise fine. Symptom: an OOM traceback through torch._inductor /
    cuda_graph.py rather than weight loading. Costs some speed."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model = patch_rope_conflict(model)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # visual benchmarks: images ride along as base64 data URLs through the
    # chat API, and the engine must be told the max images per prompt up front
    max_imgs = max((sum(len(x) for x in (r.get("images") or [])) for r in recs),
                   default=0)
    extra = {"limit_mm_per_prompt": {"image": max_imgs}} if max_imgs else {}
    llm = LLM(model=model, dtype=pick_dtype(dtype), max_model_len=max_len,
              gpu_memory_utilization=gpu_util, tensor_parallel_size=tp,
              trust_remote_code=True, enforce_eager=enforce_eager, **extra)

    if max_imgs:
        print(f"[vllm] multimodal: up to {max_imgs} images per prompt, "
              f"using llm.chat()")
        keep = [i for i, rec in enumerate(recs)
                if len(tok.encode(rec["prompt"])) + max_out < max_len - 64 - 1024 * max_imgs]
        msgs = [[{"role": "user", "content": openai_content(recs[i])}] for i in keep]
        if guided:
            sps = []
            for i in keep:
                kw = _guided_kwargs(answer_regex(recs[i]))
                if kw is None:
                    sps = None
                    break
                sps.append(SamplingParams(temperature=0.0, max_tokens=max_out, **kw))
            sp = sps or SamplingParams(temperature=0.0, max_tokens=max_out)
            print("[vllm] guided decoding " + ("ON" if sps else
                  "unavailable in this build, free text"))
        else:
            sp = SamplingParams(temperature=0.0, max_tokens=max_out)
        outs = llm.chat(msgs, sampling_params=sp)
        texts = {i: o.outputs[0].text.strip() for i, o in zip(keep, outs)}
        print(f"[vllm] generated {len(texts)}/{len(recs)} "
              f"({len(recs)-len(texts)} skipped: prompt too long)")
        return texts

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
    ap.add_argument("--backend", default="api",
                    choices=["api", "vllm", "openai", "anthropic"],
                    help="api = Gemini/Vertex; anthropic = Claude "
                         "(ANTHROPIC_API_KEY); openai = OpenAI, Kimi/Moonshot, "
                         "OpenRouter or any OpenAI-compatible endpoint "
                         "(OPENAI_API_KEY, OPENAI_BASE_URL); vllm = local "
                         "weights. Visual benchmarks need a multimodal model "
                         "on any backend")
    ap.add_argument("--max-fail", type=int, default=5,
                    help="give up on an arm after this many consecutive "
                         "failures with no success: a wrong key or an "
                         "unsupported payload fails every item, and retrying "
                         "200 of them wastes half an hour")
    ap.add_argument("--reasoning-effort", default="",
                    help="openai backend: passed through as reasoning_effort "
                         "(minimal|low|medium|high). The paper runs with "
                         "thinking disabled, so 'minimal' is the closest match "
                         "and is far cheaper; a long reasoning pass also needs "
                         "a bigger --max-out or it returns empty text")
    ap.add_argument("--tp", type=int, default=1, help="vllm tensor-parallel size")
    ap.add_argument("--max-len", type=int, default=32768, help="vllm max_model_len")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="vllm: skip CUDA-graph/torch.compile capture. Use "
                         "this when an OOM's traceback runs through "
                         "torch._inductor / cuda_graph.py rather than weight "
                         "loading -- the compile step itself needs extra "
                         "scratch memory that can tip an otherwise-fitting "
                         "model over the edge on a smaller GPU (e.g. a newer "
                         "or larger model on a T4). Costs some speed.")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16"],
                    help="vllm weights dtype. auto picks float16 on pre-Ampere "
                         "GPUs (Kaggle/Colab T4), which have no bfloat16")
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
        # A cache entry is only reusable if it answered the SAME question and
        # (b) actually succeeded -- an errored call is retried, never treated
        # as done, or fixing the key and re-running would report the same
        # failure forever with zero new calls.
        stale = failed = 0
        for rec in recs:
            k = rec.get("item_id") or rec["group_id"]
            if k in done:
                if done[k].get("err"):
                    del done[k]
                    failed += 1
                    continue
                cached, now = done[k].get("phash"), phash(rec["prompt"], rec.get("images"))
                # fall back to question text for results written before phash
                bad = (cached != now) if cached else (
                    done[k].get("questions") != rec.get("questions"))
                if bad:
                    del done[k]
                    stale += 1
        print(f"resuming: {len(done)} already done"
              + (f"  ({stale} DISCARDED — cached answers were for different "
                 f"questions; the input was rebuilt)" if stale else "")
              + (f"  ({failed} RETRYING — previously failed, not cached)"
                 if failed else ""))

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
                    phash=phash(rec["prompt"], rec.get("images")),  # resume-safety
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
                         guided=not args.no_guided,
                         dtype=args.dtype,
                         enforce_eager=args.enforce_eager) if todo else {}
        results = [done[(r.get("item_id") or r["group_id"])]
                   for r in recs if (r.get("item_id") or r["group_id"]) in done]
        for i, rec in enumerate(todo):
            results.append(score_one(rec, texts.get(i, ""),
                                     None if i in texts else "skipped:too_long"))
    else:
        if args.backend == "openai":
            client = make_openai_client()
            eff = args.reasoning_effort

            def fn(c, m, rec, mo, rpm):
                return call_openai(c, m, rec, mo, rpm, effort=eff)
        elif args.backend == "anthropic":
            client = make_anthropic_client()
            fn = call_anthropic
        else:
            client = make_client()
            fn = call
        n_img = sum(1 for r in recs if any(r.get("images") or []))
        if n_img:
            print(f"[mm] {n_img}/{len(recs)} records carry images -> "
                  f"{args.model} must be a vision model")

        fail = {"n": 0, "ok": 0}

        def work(rec):
            key = rec.get("item_id") or rec["group_id"]
            if key in done:
                return done[key]
            if fail["n"] >= args.max_fail and fail["ok"] == 0:
                # every call so far has failed: the cause is configuration
                # (key, credit, unsupported payload), not one bad item. Stop
                # instead of spending the retry budget 200 times over.
                return score_one(rec, "", "error:aborted")
            txt, err = fn(client, args.model, rec, args.max_out, args.rpm)
            if err:
                fail["n"] += 1
            else:
                fail["ok"] += 1
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
        if errs:
            kinds = Counter(r["err"].split(":", 2)[1] for r in results
                            if r.get("err") and ":" in r["err"])
            print(f"  error types: {dict(kinds)}")
            first = next((r["err"] for r in results if r.get("err")), "")
            print(f"  first error: {first[:300]}")
            if errs == len(results):
                print("  EVERY call failed -> configuration, not the data. "
                      "Check the message above: 401/403 = key, 402/quota = "
                      "credit, 400 = payload (often the images).")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
