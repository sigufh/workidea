#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real paper-style experiment with 4-GPU parallel extraction")
    parser.add_argument("--manifest", required=True, help="JSONL: {id, video, prompt}")
    parser.add_argument("--out-dir", default="runs/real_paper", help="Output root")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
        ],
    )
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU ids, comma separated")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--target-tokens", type=int, default=1024)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fullkv", "h2o", "snap", "pyramid", "vlcache", "streamingcache", "dynamic_freq_window"],
    )
    return parser.parse_args()


def load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if "id" not in row or "video" not in row or "prompt" not in row:
                raise ValueError("Each manifest row must include: id, video, prompt")
            rows.append(row)
    return rows


def infer_backend(model_id: str) -> str:
    low = model_id.lower()
    if "qwen2.5-vl" in low or "qwen-vl" in low:
        return "qwen25vl"
    if "llava" in low and "video" in low:
        return "llava_next_video"
    raise ValueError(f"Cannot infer backend for model: {model_id}")


def launch_jobs(args: argparse.Namespace, items: list[dict], gpu_ids: list[str], trace_dir: Path) -> list[Path]:
    tasks = []
    for model_id in args.models:
        backend = infer_backend(model_id)
        model_tag = model_id.replace("/", "__")
        for item in items:
            trace_path = trace_dir / f"{item['id']}__{model_tag}.npz"
            cmd = [
                "python",
                "scripts/extract_trace_vlm_hf.py",
                "--model-id",
                model_id,
                "--backend",
                backend,
                "--video",
                item["video"],
                "--prompt",
                item["prompt"],
                "--max-frames",
                str(args.max_frames),
                "--sample-fps",
                str(args.sample_fps),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--out",
                str(trace_path),
            ]
            tasks.append((cmd, trace_path))

    running: list[tuple[subprocess.Popen, Path, str, object]] = []
    done_paths: list[Path] = []
    next_task = 0

    while next_task < len(tasks) or running:
        while next_task < len(tasks) and len(running) < len(gpu_ids):
            cmd, out_path = tasks[next_task]
            gpu = gpu_ids[len(running) % len(gpu_ids)]
            env = dict(**os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = out_path.with_suffix(".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running.append((proc, out_path, gpu, logf))
            print(f"[launch] gpu={gpu} trace={out_path.name}")
            next_task += 1

        still = []
        for proc, out_path, gpu, logf in running:
            code = proc.poll()
            if code is None:
                still.append((proc, out_path, gpu, logf))
                continue
            logf.close()
            if code != 0:
                raise RuntimeError(f"Job failed on gpu {gpu}: {out_path}")
            done_paths.append(out_path)
            print(f"[done] gpu={gpu} trace={out_path.name}")
        running = still
        if running:
            time.sleep(1.0)

    return done_paths


def run_benchmark(args: argparse.Namespace, traces: list[Path], out_json: Path) -> None:
    cmd = [
        "python",
        "scripts/run_paper_benchmark.py",
        "--traces",
        *[str(p) for p in traces],
        "--target-tokens",
        str(args.target_tokens),
        "--strategies",
        *args.strategies,
        "--out",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    items = load_manifest(args.manifest)
    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided")

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    report_dir = out_dir / "reports"
    trace_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    traces = launch_jobs(args, items, gpu_ids, trace_dir)
    out_json = report_dir / f"paper_benchmark_t{args.target_tokens}.json"
    run_benchmark(args, traces, out_json)
    print(f"saved report: {out_json}")


if __name__ == "__main__":
    main()
