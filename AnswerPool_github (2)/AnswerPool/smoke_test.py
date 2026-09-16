#!/usr/bin/env python
# ============================================================================
# smoke_test.py — end-to-end check of the pipeline on SYNTHETIC data.
#
# Needs no network, no HF token, no GPU and no API key. It installs a fake
# `datasets` module that returns rows shaped exactly like each real benchmark
# (HellaSwag / GPQA Diamond / MMMU / MMMU-Pro / MathVista / ScienceQA,
# including PIL images), builds every arm, runs run_matching.py against a
# fake client that answers from the gold (with images checked), and finally
# runs compare.py. If this passes, the only thing left to go wrong on a real
# run is credentials or dataset access.
#
#   python smoke_test.py
# ============================================================================
import io
import json
import os
import random
import shutil
import subprocess
import sys
import types

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "_smoke")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def img(tag):
    im = Image.new("RGB", (64, 32), (240, 240, 240))
    ImageDraw.Draw(im).text((2, 10), tag, fill=(0, 0, 0))
    return im


# ------------------------------------------------------------ fake datasets --
def fake_rows(name, cfg, split):
    rng = random.Random(hash((name, cfg, split)) & 0xffff)
    if name == "Rowan/hellaswag":
        rows = []
        for act in ("Cooking", "Skiing", "Painting"):
            for i in range(12):
                ends = [f"{act} ending {i}-{k} {rng.random():.3f}" for k in range(4)]
                rows.append(dict(ind=len(rows), activity_label=act,
                                 ctx=f"Someone is doing {act.lower()} step {i}.",
                                 endings=ends, label=str(rng.randrange(4))))
        return rows
    if name == "Idavidrein/gpqa":
        assert cfg == "gpqa_diamond"
        rows = []
        for sub in ("Quantum Mechanics", "Organic Chemistry"):
            for i in range(12):
                rows.append({"Question": f"{sub} question {i}?",
                             "Correct Answer": f"correct {sub} {i}",
                             "Incorrect Answer 1": f"wrong1 {sub} {i}",
                             "Incorrect Answer 2": f"wrong2 {sub} {i}",
                             "Incorrect Answer 3": f"wrong3 {sub} {i}",
                             "Subdomain": sub})
        return rows
    if name == "MMMU/MMMU":
        rows = []
        for i in range(12):
            r = {"id": f"validation_{cfg}_{i}", "question_type": "multiple-choice",
                 "question": f"<image 1> What does the {cfg} figure {i} show?",
                 "options": str([f"{cfg} opt {i}-{k}" for k in range(4)]),
                 "answer": LETTERS[rng.randrange(4)], "image_1": img(f"{cfg}{i}")}
            for k in range(2, 8):
                r[f"image_{k}"] = None
            if i == 3:                       # image-valued options -> dropped
                r["options"] = str(["<image 2>", "<image 3>", "x", "y"])
            if i == 4:
                r["question_type"] = "open"  # open item -> skipped
            rows.append(r)
        return rows
    if name == "MMMU/MMMU_Pro":
        rows = []
        for sub in ("Art", "Math"):
            for i in range(12):
                if cfg == "vision":
                    rows.append({"id": f"{sub}_{i}", "subject": sub, "image": img(f"{sub}{i}"),
                                 "options": str([f"{sub} v-opt {i}-{k}" for k in range(4)]),
                                 "answer": LETTERS[rng.randrange(4)]})
                else:
                    r = {"id": f"{sub}_{i}", "subject": sub,
                         "question": f"<image 1> {sub} pro question {i}?",
                         "options": str([f"{sub} p-opt {i}-{k}" for k in range(4)]),
                         "answer": LETTERS[rng.randrange(4)], "image_1": img(f"{sub}{i}")}
                    for k in range(2, 8):
                        r[f"image_{k}"] = None
                    rows.append(r)
        return rows
    if name == "AI4Math/MathVista":
        rows = []
        for task in ("geometry problem solving", "figure question answering"):
            for i in range(12):
                ch = [f"{i*7+k}" for k in range(4)] if i % 2 else [f"{task[:3]} ans {i}-{k}" for k in range(4)]
                rows.append(dict(pid=f"{task[:3]}{i}", question_type="multi_choice",
                                 question=f"{task} question {i}?", choices=ch,
                                 answer=ch[rng.randrange(4)], decoded_image=img(f"mv{i}"),
                                 metadata=dict(task=task, category="general-vqa",
                                               skill=["arithmetic"], grade="high school")))
        rows.append(dict(pid="ff0", question_type="free_form", question="free?", choices=None,
                         answer="3", decoded_image=img("ff"), metadata=dict(task="x")))
        return rows
    if name == "derek-thomas/ScienceQA":
        rows = []
        for topic in ("weather", "chemistry", "geography"):
            for i in range(12):
                rows.append(dict(image=(img(f"sq{i}") if i % 2 == 0 else None),
                                 question=f"Which {topic} statement {i} is true?",
                                 choices=[f"{topic} choice {i}-{k}" for k in range(3)],
                                 answer=rng.randrange(3), hint=(f"hint {i}" if i % 3 else ""),
                                 topic=topic, category=f"{topic}-cat", subject="natural science",
                                 skill="s"))
        return rows
    raise KeyError(name)


def install_fake_datasets():
    m = types.ModuleType("datasets")

    def load_dataset(name, cfg=None, split=None, **kw):
        if split is None and isinstance(cfg, str) and cfg in ("validation", "test", "train"):
            split, cfg = cfg, None
        return fake_rows(name, cfg, split)

    def get_dataset_config_names(name):
        return {"MMMU/MMMU": ["Art", "Biology"]}.get(name, ["all"])
    m.load_dataset = load_dataset
    m.get_dataset_config_names = get_dataset_config_names
    sys.modules["datasets"] = m


# ------------------------------------------------------------- fake client --
class FakeModels:
    """Answers from the gold stored on the record, but only if the prompt text
    reaches it -- and, on visual records, only if the images were attached."""
    def __init__(self, gold_by_phash):
        self.gold = gold_by_phash
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        text = [c for c in contents if isinstance(c, str)][-1]
        imgs = [c for c in contents if isinstance(c, Image.Image)]
        g, want_imgs = self.gold[text]
        assert len(imgs) == want_imgs, f"expected {want_imgs} images, got {len(imgs)}"
        if len(g) == 1:
            body = g[0]
        else:
            body = "\n".join(f"{i+1}: {x}" for i, x in enumerate(g))
        return types.SimpleNamespace(text=body)


def run_matching_fake(inp, model="fake-vision"):
    import run_matching
    recs = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    gold = {r["prompt"]: (r["answer"], sum(len(x) for x in r.get("images") or []))
            for r in recs}
    fm = FakeModels(gold)
    run_matching.make_client = lambda: types.SimpleNamespace(models=fm)
    run_matching.make_config = lambda mo: None
    argv = sys.argv
    sys.argv = ["run_matching.py", "--in", inp, "--model", model, "--workers", "4"]
    try:
        run_matching.main()
    finally:
        sys.argv = argv
    return fm.calls


def sh(cmd):
    print("$", " ".join(cmd))
    r = subprocess.run([sys.executable] + cmd, cwd=WORK, text=True,
                       capture_output=True, env={**os.environ, "PYTHONPATH": ROOT})
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def build_cmd(dataset, extra, out):
    return ["-c",
            "import sys; sys.path.insert(0, %r); import smoke_test as t; t.install_fake_datasets(); "
            "sys.argv = ['build_matching.py'] + %r; import build_matching; build_matching.main()"
            % (ROOT, ["--dataset", dataset, "--out", out] + extra)]


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    install_fake_datasets()
    os.chdir(WORK)
    sys.path.insert(0, ROOT)

    cases = [
        ("hellaswag", ["--n", "3", "--distractors"]),
        ("gpqa_diamond", ["--n", "3", "--distractors"]),
        ("gpqa", ["--gpqa-config", "gpqa_diamond", "--n", "3", "--distractors"]),
        ("mmmu", ["--n", "3", "--distractors"]),
        ("mmmu_pro", ["--n", "3", "--distractors"]),
        ("mmmu_pro", ["--mmmu-pro-config", "vision", "--n", "3", "--distractors"]),
        ("mathvista", ["--n", "3", "--distractors"]),
        ("scienceqa", ["--n", "3", "--distractors"]),
        ("scienceqa", ["--n", "3", "--m", "5", "--require-image"]),   # padded, not distractor
    ]
    for i, (ds, extra) in enumerate(cases):
        tag = f"{ds}_{i}"
        # matching arm with 40% withheld
        out = sh(build_cmd(ds, extra + ["--withhold", "0.4", "--seed", "1"], f"{tag}.jsonl"))
        n_groups = int([l for l in out.splitlines() if l.startswith("wrote ")][0].split()[1])
        assert n_groups > 0, out
        # mirror-built MCQ baseline
        cfg = [x for x in extra if x.startswith("--") and x.endswith("-config")]
        cfg = sum(([c, extra[extra.index(c) + 1]] for c in cfg), [])
        sh(build_cmd(ds, cfg + ["--arm", "mcq", "--mirror", f"{tag}.jsonl", "--seed", "1"],
                     f"{tag}_mcq.jsonl"))
        # choices-only control (no images must be attached)
        sh(build_cmd(ds, extra + ["--arm", "choices_only", "--seed", "1"], f"{tag}_co.jsonl"))
        recs = [json.loads(l) for l in open(f"{tag}.jsonl") if l.strip()]
        mcq = [json.loads(l) for l in open(f"{tag}_mcq.jsonl") if l.strip()]
        co = [json.loads(l) for l in open(f"{tag}_co.jsonl") if l.strip()]
        assert len(mcq) == sum(r["n"] for r in recs), (len(mcq), sum(r["n"] for r in recs))
        assert all(not any(r["images"]) for r in co)
        # mirror alignment: identical questions, in order
        for r0 in recs:
            mine = [m for m in mcq if m["group_id"] == r0["group_id"]]
            assert [m["questions"][0] for m in sorted(mine, key=lambda x: x["q_index"])] == r0["questions"]
        # visual datasets must carry image files
        from bench_datasets import VISUAL
        if ds in VISUAL and "--require-image" not in extra and ds != "scienceqa":
            assert all(all(r["images"][q] for q in range(r["n"])) for r in recs), tag
        for r in recs:
            for paths in r["images"]:
                for p in paths:
                    assert os.path.exists(p), p
        # run all three arms with the fake model, then compare
        for f in (f"{tag}.jsonl", f"{tag}_mcq.jsonl", f"{tag}_co.jsonl"):
            calls = run_matching_fake(f)
            res = [json.loads(l) for l in open(f.replace(".jsonl", ".fake-vision.results.jsonl"))]
            assert calls == len(res)
            for r in res:
                assert r["pred"] == r["gold"], (f, r["pred"], r["gold"], r["raw"])
        sh([os.path.join(ROOT, "compare.py"), "--glob", f"{tag}*.results.jsonl"])
        print(f"OK  {tag}: {n_groups} groups, {len(mcq)} mcq items")

    # image-withheld arms on a visual benchmark: allowed, no images attached,
    # prompt says so; text-only benchmark still refuses them
    sh(build_cmd("mmmu", ["--n", "3", "--distractors", "--arm", "no_passage", "--seed", "1"],
                 "mmmu_blind.jsonl"))
    sh(build_cmd("mmmu", ["--arm", "mcq_no_passage", "--mirror", "mmmu_3.jsonl", "--seed", "1"],
                 "mmmu_blind_mcq.jsonl"))
    for f in ("mmmu_blind.jsonl", "mmmu_blind_mcq.jsonl"):
        for r in (json.loads(l) for l in open(f) if l.strip()):
            assert not any(r["images"]), f
            assert "not provided" in r["prompt"].lower(), r["prompt"]
    for f in ("mmmu_blind.jsonl", "mmmu_blind_mcq.jsonl"):
        run_matching_fake(f)
    r = subprocess.run([sys.executable] + build_cmd("hellaswag", ["--arm", "no_passage"], "x.jsonl"),
                       cwd=WORK, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode != 0 and "needs a text passage" in (r.stdout + r.stderr)
    # closed-book filter: the exclude list must bite on a visual benchmark
    import hashlib
    recs = [json.loads(l) for l in open("mmmu_3.jsonl") if l.strip()]
    from bench_datasets import norm
    h = hashlib.sha1(norm(recs[0]["questions"][0]).lower().encode()).hexdigest()[:16]
    open("blind.txt", "w").write(h + "\n")
    out = sh(build_cmd("mmmu", ["--n", "3", "--distractors", "--exclude", "blind.txt"], "mmmu_ex.jsonl"))
    assert "excluded 1 blind-solvable" in out, out
    print("OK  image-withheld arms + exclude filter")

    # openai + anthropic content assembly (no client needed)
    import run_matching
    recs = [json.loads(l) for l in open("mmmu_3.jsonl") if l.strip()]
    n_img = sum(len(x) for x in recs[0]["images"])
    c = run_matching.openai_content(recs[0])
    assert c[-1]["type"] == "text" and c[-1]["text"] == recs[0]["prompt"]
    assert sum(1 for x in c if x["type"] == "image_url") == n_img
    assert c[1]["image_url"]["url"].startswith("data:image/png;base64,")
    a = run_matching.anthropic_content(recs[0])
    assert a[-1]["text"] == recs[0]["prompt"]
    assert sum(1 for x in a if x["type"] == "image") == n_img
    assert a[1]["source"]["media_type"] == "image/png" and a[1]["source"]["data"]
    # text-only records must degrade to a plain string on both
    txt = [json.loads(l) for l in open("hellaswag_0.jsonl") if l.strip()][0]
    assert isinstance(run_matching.openai_content(txt), str)
    assert isinstance(run_matching.anthropic_content(txt), str)
    print("OK  openai + anthropic multimodal content")

    # a reasoning model that rejects temperature and max_tokens: the backend
    # must discover both from the errors and still return an answer
    class FakeChat:
        def __init__(self): self.calls = []
        def create(self, **kw):
            self.calls.append(dict(kw))
            if "temperature" in kw:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported")
            if "max_tokens" in kw:
                raise RuntimeError("Use 'max_completion_tokens' instead of 'max_tokens'")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="1: A\n2: B\n3: C"))])
    fc = FakeChat()
    cl = types.SimpleNamespace(chat=types.SimpleNamespace(completions=fc))
    run_matching._OAI.update(cap="max_tokens", temp=True)
    out, err = run_matching.call_openai(cl, "gpt-5.2", txt, 2048, 0, effort="minimal")
    assert err is None and out.startswith("1: A"), (out, err)
    assert run_matching._OAI == {"cap": "max_completion_tokens", "temp": False}
    assert fc.calls[-1]["reasoning_effort"] == "minimal"
    # the discovery is remembered: the next call is clean on the first try
    before = len(fc.calls)
    run_matching.call_openai(cl, "gpt-5.2", txt, 2048, 0)
    assert len(fc.calls) == before + 1
    print("OK  reasoning-model parameter discovery")

    # verify_wellformed.py and judge_freeform.py both call run_matching.call()
    # with a plain string built via .format() -- call() takes a rec DICT (it
    # looks up rec["images"]), so passing a raw string crashes with
    # AttributeError: 'str' object has no attribute 'get'. Both were patched
    # to wrap the string in {"prompt": ..., "images": []}; catch a regression
    # of either patch directly, without needing a live Gemini key.
    import verify_wellformed as VW
    import judge_freeform as JF
    _real_call = VW.call  # save before any test below monkeypatches VW.call
    fake_call_seen = []
    def fake_call(client, model, rec, max_out, rpm):
        assert isinstance(rec, dict), f"call() got a {type(rec).__name__}, not a dict"
        assert "images" in rec and rec["images"] == []
        fake_call_seen.append(rec["prompt"])
        return "B", None
    VW.call = fake_call
    job = ("g0", 0, "fake-model", "some prompt text", "A", 4)
    out = VW.main.__globals__  # not used; call work() indirectly instead
    # work() is a closure inside main(); test it the way main() would build it
    gid, qi, mdl, p, gold_L, m = job
    vw_rec = {"prompt": p, "images": []}
    vw_txt, vw_err = VW.call(None, mdl, vw_rec, 16, 0)
    assert vw_txt == "B" and vw_err is None
    print("OK  verify_wellformed.py call() site passes a dict, not a string")

    JF.call = fake_call
    r = {"questions": ["q?"], "raw": "some answer"}
    jf_rec = {"prompt": JF.PROMPT.format(q=r["questions"][0], a=r["raw"][:3000]), "images": []}
    jf_txt, jf_err = JF.call(None, "judge-model", jf_rec, 64, 0)
    assert jf_txt == "B" and jf_err is None
    print("OK  judge_freeform.py call() site passes a dict, not a string")

    # verify_wellformed.py must attach the RIGHT question's image (visual
    # benchmarks are PASSAGE_FREE, so the old "no passage" text fallback
    # would otherwise send an image question to the screener with nothing
    # to look at, silently reducing the visual verifier screen to a coin flip)
    import contextlib, io
    rec_v = dict(group_id="Art_0", arm="matching", withheld=[], n=2, m=8,
                questions=["What does this image show? [image 1]",
                           "Text-only question, no image"],
                candidates=["A cat", "A dog", "History fact", "Other fact",
                            "X", "Y", "Z", "W"],
                answer=["A", "C"],
                images=[["mmmu_3_images/" + os.listdir("mmmu_3_images")[0]]
                        if os.path.isdir("mmmu_3_images") and os.listdir("mmmu_3_images")
                        else [], []],
                prompt="Match each question...\n\nQUESTIONS:\n1. ...\n2. ...")
    if any(rec_v["images"]):
        open("vw_test.jsonl", "w").write(json.dumps(rec_v) + "\n")
        vw_calls = []
        class FakeVWModels:
            def generate_content(self, model, contents, config):
                vw_calls.append(contents)
                return types.SimpleNamespace(text="NO")
        VW.call = _real_call  # undo the earlier fake_call monkeypatch from the
                          # call()-signature regression test above, which
                          # hardcoded rec["images"] == [] and would otherwise
                          # silently make this test check nothing real
        VW.make_client = lambda: types.SimpleNamespace(models=FakeVWModels())
        sys.argv = ["verify_wellformed.py", "--src", "vw_test.jsonl",
                   "--models", "fake-model", "--workers", "1",
                   "--out", "vw_flagged.txt", "--detail", "vw_detail.jsonl"]
        with contextlib.redirect_stdout(io.StringIO()):
            VW.main()
        from PIL import Image as _PILImage
        for c in vw_calls:
            prompt = [x for x in c if isinstance(x, str)][-1]
            has_img = any(isinstance(x, _PILImage.Image) for x in c)
            if "What does this image show" in prompt:
                assert has_img, "verify_wellformed: image-bearing question got no image"
            else:
                assert not has_img, "verify_wellformed: image leaked to the wrong question"
        print("OK  verify_wellformed.py attaches the right question's image, none leaked")
    else:
        print("SKIP verify_wellformed.py image test (no cached mmmu_3 image on disk)")

    # anthropic: blocks are unwrapped, thinking blocks skipped
    class FakeMsgs:
        def create(self, **kw):
            assert kw["max_tokens"] and kw["model"]
            return types.SimpleNamespace(content=[
                types.SimpleNamespace(type="thinking", thinking="..."),
                types.SimpleNamespace(type="text", text=" B ")])
    out, err = run_matching.call_anthropic(
        types.SimpleNamespace(messages=FakeMsgs()), "claude-opus-5", txt, 2048, 0)
    assert (out, err) == ("B", None), (out, err)
    print("OK  anthropic backend")

    # resume must discard cached answers when images change
    p = "mathvista_6.jsonl"
    recs = [json.loads(l) for l in open(p) if l.strip()]
    h0 = run_matching.phash(recs[0]["prompt"], recs[0]["images"])
    h1 = run_matching.phash(recs[0]["prompt"], [["other.png"] + x[1:] for x in recs[0]["images"]])
    assert h0 != h1
    print("OK  phash covers images")
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
