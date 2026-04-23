#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer-level real eval on MMBench-Video")
    parser.add_argument("--manifest", required=True, help="JSONL rows: {id, video, prompt[,question_id,answer]}")
    parser.add_argument("--model-id", required=True, help="HF id or local model path")
    parser.add_argument("--backend", default="qwen25vl", choices=["qwen25vl", "llava_next_video"])
    parser.add_argument("--questions", default="data/MMBench-Video-10G/MMBench-Video_q.json")
    parser.add_argument("--answers", default="data/MMBench-Video-10G/MMBench-Video_a.json")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--out", required=True, help="Output json path")
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def sample_video_frames(path: str, sample_fps: float, max_frames: int) -> list[np.ndarray]:
    vr = VideoReader(path, ctx=cpu(0))
    native_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 25.0
    stride = max(1, int(round(native_fps / max(sample_fps, 1e-3))))
    frame_idx = list(range(0, len(vr), stride))
    if len(frame_idx) > max_frames:
        lin = np.linspace(0, len(frame_idx) - 1, num=max_frames, dtype=np.int64)
        frame_idx = [frame_idx[int(i)] for i in lin]
    batch = vr.get_batch(frame_idx).asnumpy()
    return [frame for frame in batch]


def load_processor_and_model(model_id: str, backend: str, dtype_name: str):
    dtype = resolve_dtype(dtype_name)
    if backend == "qwen25vl":
        from transformers import (
            AutoTokenizer,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )

        image_processor = Qwen2VLImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        video_processor = Qwen2VLVideoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        chat_template = None
        chat_template_path = Path(model_id) / "chat_template.json"
        if chat_template_path.exists():
            raw = chat_template_path.read_text(encoding="utf-8")
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and isinstance(obj.get("chat_template"), str):
                    chat_template = obj["chat_template"]
                else:
                    chat_template = raw
            except json.JSONDecodeError:
                chat_template = raw
        processor = Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return processor, model

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    return processor, model


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def first_int(s: str) -> int | None:
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    return int(m.group(0))


def load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def build_gt_map(question_json: str, answer_json: str) -> dict[tuple[str, str], dict]:
    q_rows = json.load(open(question_json, "r", encoding="utf-8"))
    a_rows = json.load(open(answer_json, "r", encoding="utf-8"))
    answer_by_qid = {r["question_id"]: r["answer"] for r in a_rows}
    out = {}
    for q in q_rows:
        key = (q["video_name"], normalize_text(q["question"]))
        out[key] = {
            "question_id": q["question_id"],
            "answer": answer_by_qid.get(q["question_id"], ""),
        }
    return out


def main() -> None:
    args = parse_args()
    items = load_manifest(args.manifest)
    gt_map = build_gt_map(args.questions, args.answers)

    processor, model = load_processor_and_model(args.model_id, args.backend, args.dtype)
    model.to(args.device)
    model.eval()

    rows = []
    total_gen_seconds = 0.0
    total_new_tokens = 0
    with torch.no_grad():
        for item in items:
            video_path = item["video"]
            prompt = item["prompt"]
            video_name = Path(video_path).stem
            manifest_gt = item.get("answer")
            manifest_qid = item.get("question_id")
            if manifest_gt is not None and str(manifest_gt).strip() != "":
                gt = {
                    "question_id": str(manifest_qid) if manifest_qid is not None else "",
                    "answer": str(manifest_gt),
                }
            else:
                gt = gt_map.get((video_name, normalize_text(prompt)))
                if gt is None:
                    raise ValueError(f"Cannot find GT for video={video_name}, prompt={prompt}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "nframes": args.max_frames,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if args.backend == "qwen25vl":
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                frames = sample_video_frames(video_path, args.sample_fps, args.max_frames)
                try:
                    inputs = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
                except TypeError:
                    inputs = processor(text=[text], videos=frames, return_tensors="pt", padding=True)
            model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            t0 = time.perf_counter()
            generated = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            gen_seconds = time.perf_counter() - t0
            prompt_len = int(model_inputs["input_ids"].shape[1])
            out_ids = generated[0, prompt_len:]
            pred = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            new_tokens = int(out_ids.shape[0])
            total_gen_seconds += gen_seconds
            total_new_tokens += new_tokens
            gt_answer = gt["answer"].strip()

            pred_n = normalize_text(pred)
            gt_n = normalize_text(gt_answer)
            exact_match = int(pred_n == gt_n and len(gt_n) > 0)
            contains_match = int((gt_n in pred_n or pred_n in gt_n) and len(gt_n) > 0 and len(pred_n) > 0)
            pred_i = first_int(pred)
            gt_i = first_int(gt_answer)
            numeric_match = int(pred_i is not None and gt_i is not None and pred_i == gt_i)
            if gt_i is not None:
                task_match = numeric_match
            else:
                task_match = max(exact_match, contains_match)

            rows.append(
                {
                    "id": item["id"],
                    "video": video_path,
                    "question_id": gt["question_id"],
                    "question": prompt,
                    "pred": pred,
                    "gt": gt_answer,
                    "gen_seconds": float(gen_seconds),
                    "new_tokens": new_tokens,
                    "exact_match": exact_match,
                    "contains_match": contains_match,
                    "numeric_match": numeric_match,
                    "task_match": int(task_match),
                }
            )

    n = max(1, len(rows))
    summary = {
        "num_samples": len(rows),
        "exact_match_acc": float(sum(r["exact_match"] for r in rows) / n),
        "contains_match_acc": float(sum(r["contains_match"] for r in rows) / n),
        "numeric_match_acc": float(sum(r["numeric_match"] for r in rows) / n),
        "task_acc": float(sum(r["task_match"] for r in rows) / n),
        "avg_gen_seconds": float(total_gen_seconds / n),
        "avg_new_tokens": float(total_new_tokens / n),
        "tokens_per_second": float(total_new_tokens / total_gen_seconds) if total_gen_seconds > 1e-8 else 0.0,
    }

    out = {
        "meta": {
            "manifest": args.manifest,
            "model_id": args.model_id,
            "backend": args.backend,
            "max_frames": args.max_frames,
            "sample_fps": args.sample_fps,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": summary,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved real eval report: {out_path}")


if __name__ == "__main__":
    main()
