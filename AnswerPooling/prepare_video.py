#!/usr/bin/env python
# ============================================================================
# prepare_video.py -- sample a fixed number of frames per video, once, and
# write the index that build_matching.py --frames-index reads.
#
#   Video-MME (lmms-lab/Video-MME): the videos are 20 zip chunks in the HF
#   repo (about 100 GB). --download fetches and extracts them into
#   --video-dir; otherwise point --video-dir at a folder that already holds
#   the mp4 files (searched recursively, matched by file stem = videoID).
#
#   LVBench (THUDM/LVBench): videos come from YouTube. Run the benchmark's
#   scripts/download.sh (or yt-dlp on every `key` of video_info.meta.jsonl)
#   into --video-dir first; files are matched by stem = key.
#
#   python -m AnswerPooling.prepare_video --dataset videomme --video-dir videomme_videos --download --frames 16
#   python -m AnswerPooling.prepare_video --dataset lvbench  --video-dir lvbench_videos --frames 32
#
# Frames are spread uniformly over the video, resized so the longer side is
# --max-side pixels, saved as JPEG under <dataset>_frames/<key>/, and the
# index json maps key -> [paths] in time order.
# ============================================================================
import argparse
import json
import os
import zipfile

import cv2


def video_keys(dataset):
    if dataset == "videomme":
        from datasets import load_dataset
        return sorted({str(r["videoID"]) for r in load_dataset("lmms-lab/Video-MME", split="test")})
    from huggingface_hub import hf_hub_download
    mp = hf_hub_download("THUDM/LVBench", "video_info.meta.jsonl", repo_type="dataset")
    keys = []
    for line in open(mp, encoding="utf-8"):
        line = line.strip()
        if line:
            keys.append(str(json.loads(line)["key"]))
    return sorted(set(keys))


def download_videomme(video_dir):
    """Each zip chunk is downloaded into video_dir itself (not the HF cache,
    which would hold a second 100 GB copy), extracted, and deleted."""
    from huggingface_hub import hf_hub_download
    os.makedirs(video_dir, exist_ok=True)
    for i in range(1, 21):
        name = f"videos_chunked_{i:02d}.zip"
        marker = os.path.join(video_dir, f".{name}.done")
        if os.path.exists(marker):
            continue
        print(f"downloading {name}")
        zp = hf_hub_download("lmms-lab/Video-MME", name, repo_type="dataset",
                             local_dir=video_dir)
        with zipfile.ZipFile(zp) as z:
            z.extractall(video_dir)
        os.remove(zp)
        open(marker, "w").close()
        print(f"  extracted and removed {name}")


def find_videos(video_dir):
    exts = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    out = {}
    for root, _, files in os.walk(video_dir):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                out.setdefault(stem, os.path.join(root, fn))
    return out


def sample_frames(path, n, max_side, out_dir):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    os.makedirs(out_dir, exist_ok=True)
    idxs = [int((k + 0.5) * total / n) for k in range(n)]
    paths = []
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        p = os.path.join(out_dir, f"{k:03d}.jpg")
        cv2.imwrite(p, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        paths.append(p)
    cap.release()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["videomme", "lvbench"])
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--download", action="store_true", help="videomme: fetch the HF zip chunks")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--frames-dir", default="", help="where the frames go, default <dataset>_frames next to the videos")
    ap.add_argument("--out", default="", help="index json, default <frames-dir>.json")
    a = ap.parse_args()
    frame_root = a.frames_dir or os.path.join(os.path.dirname(a.video_dir.rstrip("/")) or ".",
                                              f"{a.dataset}_frames")
    out_index = a.out or frame_root.rstrip("/") + ".json"

    if a.download and a.dataset == "videomme":
        download_videomme(a.video_dir)
    keys = video_keys(a.dataset)
    have = find_videos(a.video_dir)
    print(f"{len(keys)} videos in the benchmark, {sum(k in have for k in keys)} found under {a.video_dir}")

    index = {}
    if os.path.exists(out_index):
        index = json.load(open(out_index, encoding="utf-8"))
    missing, done = [], 0
    for key in keys:
        if key in index and all(os.path.exists(p) for p in index[key]):
            continue
        if key not in have:
            missing.append(key)
            continue
        paths = sample_frames(have[key], a.frames, a.max_side,
                              os.path.join(frame_root, key))
        if paths:
            index[key] = paths
            done += 1
        else:
            missing.append(key)
        if done % 50 == 0 and done:
            json.dump(index, open(out_index, "w", encoding="utf-8"))
            print(f"  {done} videos sampled")
    json.dump(index, open(out_index, "w", encoding="utf-8"))
    print(f"-> {out_index}: {len(index)} videos with {a.frames} frames each, "
          f"{len(missing)} missing or unreadable")
    if missing:
        miss_path = frame_root.rstrip("/") + "_missing.txt"
        with open(miss_path, "w") as f:
            f.write("\n".join(missing) + "\n")
        print(f"   missing keys listed in {miss_path}")


if __name__ == "__main__":
    main()
