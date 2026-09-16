#!/usr/bin/env python
# ============================================================================
# models.py — which backend, model id and flags each model needs.
#
#   python models.py                 # list everything and what is configured
#   python models.py --resolve gpt-5.2
#
# run_benchmarks.sh takes MODEL=<key or raw id> and asks this file for the
# rest, so the driver does not hard-code provider quirks.
#
# Model ids and endpoints move. Verified against provider docs in September
# 2026; if a call 404s, check the provider console and edit the table.
# ============================================================================
import argparse
import os

# key -> (backend, api id, base_url env value or "", extra run_matching flags, vision)
REGISTRY = {
    # ---- Anthropic (ANTHROPIC_API_KEY) ------------------------------------
    "opus-5":      ("anthropic", "claude-opus-5", "", "", True),
    "sonnet-5":    ("anthropic", "claude-sonnet-5", "", "", True),
    "fable-5.1":   ("anthropic", "claude-fable-5-1", "", "", True),
    "haiku-4.5":   ("anthropic", "claude-haiku-4-5-20251001", "", "", True),

    # ---- OpenAI (OPENAI_API_KEY) ------------------------------------------
    # GPT-5 models are reasoning models: they reject temperature and rename the
    # token cap. run_matching.py discovers both from the first error. Keep
    # reasoning_effort low and --max-out high, or the answer arrives empty.
    "gpt-5":       ("openai", "gpt-5", "", "--reasoning-effort minimal --max-out 4096", True),
    "gpt-5.2":     ("openai", "gpt-5.2", "", "--reasoning-effort minimal --max-out 4096", True),
    "gpt-5.2-chat": ("openai", "gpt-5.2-chat-latest", "", "--max-out 2048", True),

    # ---- Moonshot / Kimi (OPENAI_API_KEY from platform.kimi.ai) -----------
    # OpenAI-compatible; K3 takes text and images.
    # key goes in MOONSHOT_API_KEY (or KIMI_API_KEY), not OPENAI_API_KEY,
    # so an OpenAI key and a Moonshot key can coexist
    "kimi-k3":     ("openai", "kimi-k3", "https://api.moonshot.ai/v1",
                    "--max-out 4096 --workers 2", True),
    "kimi-k2.6":   ("openai", "kimi-k2.6", "https://api.moonshot.ai/v1",
                    "--max-out 2048 --workers 2", False),

    # ---- Google (GOOGLE_API_KEY or VERTEX_KEY) ----------------------------
    # gemini-2.5-flash/-pro are retired for new-user accounts as of the API's
    # own 404 message (2026); 3.6 Flash is Google's current free-tier model.
    "gemini-flash": ("api", "gemini-3.6-flash", "", "", True),
    "gemini-pro":   ("api", "gemini-3.1-pro", "", "--max-out 4096", True),
    "gemini-2.5-pro": ("api", "gemini-2.5-pro", "", "--max-out 4096", True),  # may 404 for new accounts

    # GitHub Models (free tier that used to live here) was permanently retired
    # 2026-07-30: https://github.blog/changelog/2026-07-30-github-models-is-now-retired/
    # The endpoint returns 410 github_models_retirement_brownout for every
    # call now. Do not re-add entries pointing at models.github.ai/inference.

    # ---- local weights via vLLM -------------------------------------------
    "qwen3-8b":    ("vllm", "Qwen/Qwen3-8B", "", "--max-len 16384", False),
    "qwen2.5-vl-7b": ("vllm", "Qwen/Qwen2.5-VL-7B-Instruct", "",
                      "--max-len 8192 --tp 2", True),
}

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "api": "GOOGLE_API_KEY", "vllm": ""}
FREE_KEYS = {"gpt-4o-free", "gpt-4.1-free", "llama-vision-free"}


def resolve(name):
    """-> dict for a registry key, or a best guess for a raw model id."""
    if name in REGISTRY:
        b, mid, base, extra, vis = REGISTRY[name]
    else:
        low = name.lower()
        if low.startswith("claude"):
            b, base, extra, vis = "anthropic", "", "", True
        elif low.startswith(("gpt-5", "o3", "o4")):
            b, base = "openai", ""
            extra, vis = "--reasoning-effort minimal --max-out 4096", True
        elif low.startswith("kimi"):
            b, base, extra, vis = "openai", "https://api.moonshot.ai/v1", "--max-out 4096", True
        elif low.startswith("gemini"):
            b, base, extra, vis = "api", "", "", True
        elif "/" in name:
            b, base, extra, vis = "vllm", "", "", False
        else:
            b, base, extra, vis = "openai", "", "", False
        mid = name
    env = KEY_ENV[b]
    if b == "openai" and base:
        low_b = base.lower()
        if "moonshot" in low_b or "kimi" in low_b:
            env = "MOONSHOT_API_KEY"
        elif "openrouter" in low_b:
            env = "OPENROUTER_API_KEY"
        elif "models.github.ai" in low_b:
            env = "GITHUB_TOKEN"
    return dict(key=name, backend=b, model=mid, base_url=base, extra=extra,
                vision=vis, key_env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve", default="")
    ap.add_argument("--shell", action="store_true",
                    help="print shell assignments for run_benchmarks.sh")
    a = ap.parse_args()
    if a.resolve:
        r = resolve(a.resolve)
        if a.shell:
            # shlex.quote: MODEL_EXTRA holds spaces, and an unquoted value
            # makes the shell try to run its second word as a command
            import shlex
            print(f"BACKEND={shlex.quote(r['backend'])}")
            print(f"MODEL_ID={shlex.quote(r['model'])}")
            print(f"MODEL_BASE_URL={shlex.quote(r['base_url'])}")
            print(f"MODEL_EXTRA={shlex.quote(r['extra'])}")
            print(f"MODEL_KEY_ENV={shlex.quote(r['key_env'])}")
            return
        for k, v in r.items():
            print(f"{k:10} {v}")
        if r["key_env"] and not os.environ.get(r["key_env"]):
            print(f"\n!! {r['key_env']} is not set")
        return
    print(f"{'key':16} {'backend':10} {'model id':34} vision  key set")
    for k in REGISTRY:
        r = resolve(k)
        ok = "-" if not r["key_env"] else ("yes" if os.environ.get(r["key_env"]) else "NO")
        print(f"{k:16} {r['backend']:10} {r['model']:34} "
              f"{'yes' if r['vision'] else 'no ':6}  {ok}")
    print("\nVisual benchmarks (mmmu, mmmu_pro, mathvista, scienceqa) need a "
          "vision model.\nA raw model id also works; the backend is guessed "
          "from its prefix.")


if __name__ == "__main__":
    main()
