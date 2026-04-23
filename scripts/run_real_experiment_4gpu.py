#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    parser.add_argument("--target-tokens", type=int, nargs="+", default=[1024])
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Forwarded to extractor. Use eager for stable attention extraction.",
    )
    parser.add_argument("--strict-attn", action="store_true", help="Fail extraction if attention is all zero.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fullkv", "h2o", "snap", "pyramid", "vlcache", "streamingcache", "dynamic_freq_window"],
    )
    parser.add_argument(
        "--run-answer-eval",
        action="store_true",
        help="If manifest rows contain answers, also run answer-level eval and LLM judge scoring.",
    )
    parser.add_argument("--judge-backend", default="openai", choices=["local", "openai"])
    parser.add_argument("--judge-model", default="qwen3-max", help="Judge model name for cloud backend or local path.")
    parser.add_argument(
        "--judge-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL for cloud judge backend.",
    )
    parser.add_argument("--judge-api-key", default="", help="API key for cloud judge backend.")
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
                "--attn-impl",
                args.attn_impl,
                "--out",
                str(trace_path),
            ]
            if args.strict_attn:
                cmd.append("--strict-attn")
            tasks.append((cmd, trace_path))

    running: list[tuple[subprocess.Popen, Path, str, object, float, Path]] = []
    free_gpus: list[str] = list(gpu_ids)
    done_paths: list[Path] = []
    next_task = 0
    total_tasks = len(tasks)
    start_all_ts = time.time()
    tick = 0

    def print_progress_line() -> None:
        nonlocal tick
        tick += 1
        done = len(done_paths)
        running_n = len(running)
        queued = total_tasks - next_task
        pct = (100.0 * done / total_tasks) if total_tasks > 0 else 100.0
        bar_w = 28
        filled = int(round((done / total_tasks) * bar_w)) if total_tasks > 0 else bar_w
        bar = "#" * filled + "-" * (bar_w - filled)
        spinner = "|/-\\"[tick % 4]
        elapsed = time.time() - start_all_ts
        task_parts = []
        for _p, out_path, gpu, _logf, st, progress_path in running:
            step_txt = "init"
            if progress_path.exists():
                try:
                    info = json.loads(progress_path.read_text(encoding="utf-8"))
                    step = int(info.get("step", 0))
                    total = int(info.get("total_steps", 0))
                    phase = str(info.get("phase", ""))
                    if total > 0:
                        step_txt = f"{phase}:{step}/{total}"
                    else:
                        step_txt = phase or "run"
                except Exception:
                    step_txt = "run"
            task_parts.append(f"gpu{gpu}:{out_path.name}:{step_txt}:{(time.time() - st):.0f}s")

        task_tail = " | " + " ; ".join(task_parts) if task_parts else ""
        line = (
            f"\r[{spinner}] |{bar}| {done}/{total_tasks} "
            f"({pct:5.1f}%) running={running_n} queued={queued} elapsed={elapsed:6.1f}s"
            f"{task_tail}"
        )
        if sys.stdout.isatty():
            print(line, end="", flush=True)
        else:
            # Fallback for non-interactive output.
            print(line.strip())

    while next_task < len(tasks) or running:
        while next_task < len(tasks) and free_gpus:
            cmd, out_path = tasks[next_task]
            gpu = free_gpus.pop(0)
            env = dict(**os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = out_path.with_suffix(".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            start_ts = time.time()
            progress_path = out_path.with_suffix(out_path.suffix + ".progress.json")
            running.append((proc, out_path, gpu, logf, start_ts, progress_path))
            print(f"[launch] gpu={gpu} trace={out_path.name}")
            next_task += 1

        still = []
        for proc, out_path, gpu, logf, start_ts, progress_path in running:
            code = proc.poll()
            if code is None:
                still.append((proc, out_path, gpu, logf, start_ts, progress_path))
                continue
            logf.close()
            free_gpus.append(gpu)
            if code != 0:
                # Stop all remaining workers to avoid leaving partial/corrupted outputs.
                for p2, out2, _g2, logf2, _st2, progress2 in still:
                    try:
                        p2.terminate()
                    except Exception:
                        pass
                    try:
                        p2.wait(timeout=5)
                    except Exception:
                        try:
                            p2.kill()
                        except Exception:
                            pass
                    try:
                        logf2.close()
                    except Exception:
                        pass
                    # Best-effort cleanup of potentially half-written traces.
                    try:
                        if out2.exists():
                            out2.unlink()
                    except Exception:
                        pass
                    try:
                        progress2.unlink(missing_ok=True)
                    except Exception:
                        pass
                if sys.stdout.isatty():
                    print()
                raise RuntimeError(f"Job failed on gpu {gpu}: {out_path}")
            done_paths.append(out_path)
            elapsed = time.time() - start_ts
            if sys.stdout.isatty():
                print()
            print(f"[done] gpu={gpu} trace={out_path.name} elapsed={elapsed:.1f}s progress={len(done_paths)}/{total_tasks}")
        running = still
        if running:
            print_progress_line()
        if running:
            time.sleep(1.0)

    if sys.stdout.isatty():
        print()
    return done_paths


def run_benchmark(args: argparse.Namespace, traces: list[Path], out_json: Path) -> None:
    cmd = [
        "python",
        "scripts/run_paper_benchmark.py",
        "--traces",
        *[str(p) for p in traces],
        "--target-tokens",
        *[str(x) for x in args.target_tokens],
        "--important",
        "auto",
        "--strategies",
        *args.strategies,
        "--out",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)


def manifest_has_answers(items: list[dict]) -> bool:
    return all(str(item.get("answer", "")).strip() for item in items)


def run_answer_eval(args: argparse.Namespace, items: list[dict], report_dir: Path) -> None:
    if not manifest_has_answers(items):
        print("[skip] manifest has no ground-truth answers; skip answer-level eval")
        return

    api_key = args.judge_api_key.strip() or os.environ.get("DASHSCOPE_API_KEY", "")
    if args.judge_backend == "openai" and not api_key:
        raise RuntimeError("Qwen3 cloud judge requires --judge-api-key or DASHSCOPE_API_KEY")

    for model_id in args.models:
        backend = infer_backend(model_id)
        model_tag = model_id.replace("/", "__")
        pred_json = report_dir / f"mmbench_video_real_eval_{model_tag}.json"
        judged_json = report_dir / f"mmbench_video_real_eval_{model_tag}_{args.judge_model.replace('/', '__')}_judged.json"

        eval_cmd = [
            "python",
            "scripts/run_mmbench_video_real_eval.py",
            "--manifest",
            args.manifest,
            "--model-id",
            model_id,
            "--backend",
            backend,
            "--max-frames",
            str(args.max_frames),
            "--sample-fps",
            str(args.sample_fps),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--out",
            str(pred_json),
        ]
        subprocess.run(eval_cmd, check=True)

        judge_cmd = [
            "python",
            "scripts/judge_mmbench_video_official_style.py",
            "--pred-json",
            str(pred_json),
            "--judge-backend",
            args.judge_backend,
            "--openai-model",
            args.judge_model,
            "--openai-base-url",
            args.judge_base_url,
            "--out",
            str(judged_json),
        ]
        env = dict(os.environ)
        if api_key:
            env["DASHSCOPE_API_KEY"] = api_key
        subprocess.run(judge_cmd, check=True, env=env)
        print(f"[saved] answer eval report: {pred_json.name}")
        print(f"[saved] judged report: {judged_json.name}")


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
    if len(args.target_tokens) == 1:
        suffix = f"t{args.target_tokens[0]}"
    else:
        suffix = "multi_budget"
    out_json = report_dir / f"paper_benchmark_{suffix}.json"
    run_benchmark(args, traces, out_json)
    if args.run_answer_eval:
        run_answer_eval(args, items, report_dir)
    print(f"saved report: {out_json}")


if __name__ == "__main__":
    main()
