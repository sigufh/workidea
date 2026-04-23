#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from kvbench.registry import build_strategy
from kvbench.types import CompressionContext, KVCacheState, LayerKV, TokenMeta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real answer-level evaluation with online KV strategy compression")
    parser.add_argument("--manifest", required=True, help="JSONL rows: {id, video, prompt[,question_id,answer]}")
    parser.add_argument("--model-id", required=True, help="HF/local model path")
    parser.add_argument("--backend", default="qwen25vl", choices=["qwen25vl"], help="Currently supports qwen25vl")
    parser.add_argument("--questions", default="data/MMBench-Video-10G/MMBench-Video_q.json")
    parser.add_argument("--answers", default="data/MMBench-Video-10G/MMBench-Video_a.json")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fullkv", "h2o", "snap", "pyramid", "vlcache", "streamingcache", "dynamic_freq_window"],
    )
    parser.add_argument("--target-tokens", type=int, default=1024)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--judge-backend", default="openai", choices=["openai", "local"])
    parser.add_argument("--judge-model", default="qwen3-max")
    parser.add_argument("--judge-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


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


def load_qwen_processor_and_model(model_id: str, dtype_name: str):
    dtype = resolve_dtype(dtype_name)
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


def build_messages(video_path: str, prompt: str, max_frames: int):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": max_frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def qwen_inputs(processor, messages):
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    return inputs


def _layer_attn_last_query(attn_l) -> np.ndarray:
    if attn_l is None:
        return np.zeros((1, 1), dtype=np.float32)
    t = attn_l.detach().float().cpu()
    if t.ndim == 4:
        return t[0, :, -1, :].numpy().astype(np.float32)
    if t.ndim == 3:
        return t[:, -1, :].numpy().astype(np.float32)
    if t.ndim == 2:
        return t.numpy().astype(np.float32)
    return np.zeros((1, 1), dtype=np.float32)


def _as_legacy_past(past_key_values):
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    if hasattr(past_key_values, "__iter__"):
        try:
            return tuple(past_key_values)
        except TypeError:
            pass
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return tuple((k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache))
    return past_key_values


def _to_model_cache(past_key_values, model_config):
    from transformers.cache_utils import DynamicCache

    if past_key_values is None:
        return None
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    legacy = _as_legacy_past(past_key_values)
    return DynamicCache(ddp_cache_data=legacy, config=model_config)


def _kv_state_from_past(
    past_key_values,
    attentions,
    token_meta: list[TokenMeta],
    current_step: int,
) -> KVCacheState:
    layers: list[LayerKV] = []
    for li, kv in enumerate(past_key_values):
        k = kv[0][0].detach().float().cpu().numpy().astype(np.float32)
        v = kv[1][0].detach().float().cpu().numpy().astype(np.float32)
        attn = None
        if attentions is not None and li < len(attentions) and attentions[li] is not None:
            attn = _layer_attn_last_query(attentions[li])
        if attn is None:
            attn = np.zeros((k.shape[0], k.shape[1]), dtype=np.float32)
        if attn.shape[1] != k.shape[1]:
            # align token dim
            if attn.shape[1] > k.shape[1]:
                attn = attn[:, : k.shape[1]]
            else:
                pad = np.zeros((attn.shape[0], k.shape[1] - attn.shape[1]), dtype=np.float32)
                attn = np.concatenate([attn, pad], axis=1)
        layers.append(
            LayerKV(
                keys=k,
                values=v,
                attention_scores=attn,
                importance_ema=attn.copy(),
            )
        )
    return KVCacheState(layers=layers, token_meta=token_meta, current_step=current_step)


def _past_from_state(state: KVCacheState, template_past, device: str):
    template_past = list(_as_legacy_past(template_past))
    out = []
    for layer, tpl in zip(state.layers, template_past):
        ref_k = tpl[0]
        ref_v = tpl[1]
        k = torch.from_numpy(layer.keys).to(device=device, dtype=ref_k.dtype).unsqueeze(0)
        v = torch.from_numpy(layer.values).to(device=device, dtype=ref_v.dtype).unsqueeze(0)
        if len(tpl) == 2:
            out.append((k, v))
        else:
            extras = list(tpl[2:])
            out.append((k, v, *extras))
    return tuple(out)


def _run_one_item_with_strategy(
    model,
    processor,
    strategy_name: str,
    target_tokens: int,
    video_path: str,
    prompt: str,
    max_frames: int,
    max_new_tokens: int,
) -> tuple[str, float, int]:
    strategy = build_strategy(strategy_name)
    messages = build_messages(video_path, prompt, max_frames=max_frames)
    inputs = qwen_inputs(processor, messages)
    model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # initialize token metadata from prompt length
    prompt_len = int(model_inputs["input_ids"].shape[1])
    token_meta = [
        TokenMeta(
            token_id=int(model_inputs["input_ids"][0, i].item()),
            timestep=i,
            modality="text",
            is_sink=(i < 8),
            is_special_memory=False,
        )
        for i in range(prompt_len)
    ]
    next_timestep = prompt_len
    generated_ids: list[int] = []
    total_gen_seconds = 0.0

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(use_cache=True, output_attentions=True, **model_inputs)
        total_gen_seconds += time.perf_counter() - t0
        past = _as_legacy_past(outputs.past_key_values)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(int(next_token.item()))
        token_meta.append(
            TokenMeta(
                token_id=int(next_token.item()),
                timestep=next_timestep,
                modality="text",
                is_sink=False,
                is_special_memory=False,
            )
        )
        next_timestep += 1

        # compress after prefill
        state = _kv_state_from_past(past, outputs.attentions, token_meta, current_step=0)
        ctx = CompressionContext(target_tokens=target_tokens, current_step=0, attention_history=None)
        new_state, _ = strategy.apply(state, ctx)
        past = _past_from_state(new_state, past, str(model.device))
        token_meta = new_state.token_meta

        for step in range(max(0, max_new_tokens - 1)):
            t0 = time.perf_counter()
            outputs = model(
                input_ids=next_token.to(model.device),
                use_cache=True,
                past_key_values=_to_model_cache(past, model.config),
                output_attentions=True,
            )
            total_gen_seconds += time.perf_counter() - t0
            past = _as_legacy_past(outputs.past_key_values)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(int(next_token.item()))
            token_meta.append(
                TokenMeta(
                    token_id=int(next_token.item()),
                    timestep=next_timestep,
                    modality="text",
                    is_sink=False,
                    is_special_memory=False,
                )
            )
            next_timestep += 1

            state = _kv_state_from_past(past, outputs.attentions, token_meta, current_step=step + 1)
            ctx = CompressionContext(target_tokens=target_tokens, current_step=step + 1, attention_history=None)
            new_state, _ = strategy.apply(state, ctx)
            past = _past_from_state(new_state, past, str(model.device))
            token_meta = new_state.token_meta

    pred = processor.tokenizer.decode(torch.tensor(generated_ids), skip_special_tokens=True).strip()
    return pred, float(total_gen_seconds), int(len(generated_ids))


def evaluate_strategy(
    args: argparse.Namespace,
    model,
    processor,
    strategy_name: str,
    items: list[dict],
    gt_map: dict[tuple[str, str], dict],
) -> dict:
    rows = []
    total_gen_seconds = 0.0
    total_new_tokens = 0
    total = len(items)
    for idx, item in enumerate(items, start=1):
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

        pred, gen_seconds, new_tokens = _run_one_item_with_strategy(
            model=model,
            processor=processor,
            strategy_name=strategy_name,
            target_tokens=args.target_tokens,
            video_path=video_path,
            prompt=prompt,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
        )
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
        task_match = numeric_match if gt_i is not None else max(exact_match, contains_match)

        rows.append(
            {
                "id": item["id"],
                "video": video_path,
                "question_id": gt["question_id"],
                "question": prompt,
                "pred": pred,
                "gt": gt_answer,
                "strategy": strategy_name,
                "target_tokens": args.target_tokens,
                "gen_seconds": gen_seconds,
                "new_tokens": new_tokens,
                "exact_match": exact_match,
                "contains_match": contains_match,
                "numeric_match": numeric_match,
                "task_match": int(task_match),
            }
        )
        print(
            f"[progress] strategy={strategy_name} {idx}/{total} "
            f"task_acc={sum(r['task_match'] for r in rows)/max(1, len(rows)):.4f} "
            f"avg_gen_s={total_gen_seconds/max(1, len(rows)):.2f}",
            flush=True,
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
    return {"summary": summary, "rows": rows}


def run_judge(args: argparse.Namespace, pred_json: Path, out_json: Path) -> None:
    cmd = [
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
        str(out_json),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.backend != "qwen25vl":
        raise ValueError("Only qwen25vl backend is currently supported for strategy real eval.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(args.manifest)
    gt_map = build_gt_map(args.questions, args.answers)

    processor, model = load_qwen_processor_and_model(args.model_id, args.dtype)
    model.to(args.device)
    model.eval()

    overall = {"meta": vars(args), "strategies": []}
    for strategy_name in args.strategies:
        print(f"[run] strategy={strategy_name} target_tokens={args.target_tokens}")
        result = evaluate_strategy(args, model, processor, strategy_name, items, gt_map)
        pred_json = out_dir / f"strategy_real_eval_{strategy_name}_t{args.target_tokens}.json"
        pred_json.write_text(
            json.dumps(
                {
                    "meta": {
                        "manifest": args.manifest,
                        "model_id": args.model_id,
                        "strategy": strategy_name,
                        "target_tokens": args.target_tokens,
                        "max_frames": args.max_frames,
                        "sample_fps": args.sample_fps,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "summary": result["summary"],
                    "rows": result["rows"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        judged_json = out_dir / f"strategy_real_eval_{strategy_name}_t{args.target_tokens}_judged.json"
        run_judge(args, pred_json, judged_json)
        judged = json.loads(judged_json.read_text(encoding="utf-8"))
        overall["strategies"].append(
            {
                "strategy": strategy_name,
                "target_tokens": args.target_tokens,
                "task_acc": result["summary"]["task_acc"],
                "judge_score_mean": judged["summary"].get("judge_score_mean", 0.0),
                "judge_score_norm": judged["summary"].get("judge_score_norm", 0.0),
                "pred_json": str(pred_json),
                "judged_json": str(judged_json),
            }
        )
        print(
            f"[done] strategy={strategy_name} task_acc={result['summary']['task_acc']:.4f} "
            f"judge_norm={judged['summary'].get('judge_score_norm', 0.0):.4f}"
        )

    overall["strategies"].sort(key=lambda x: x["judge_score_norm"], reverse=True)
    rank_json = out_dir / f"strategy_real_eval_ranking_t{args.target_tokens}.json"
    rank_json.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved strategy real-eval ranking: {rank_json}")


if __name__ == "__main__":
    main()
