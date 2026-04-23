#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


JUDGE_PROMPT = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct groundtruth answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, based solely on the semantic similarity between the groundtruth and the candidate answer, disregarding any grammatical differences.

A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
A rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.

Return strict JSON in the format {{"score": 0, "reason": "short reason"}} where score is one of 0, 1, 2, 3.

Question: {question}
Groundtruth answer: {reference}
Candidate answer: {prediction}
Your response:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style (LLM judge) scoring for MMBench-Video predictions")
    parser.add_argument("--pred-json", required=True, help="Prediction report json from run_mmbench_video_real_eval.py")
    parser.add_argument("--judge-backend", default="local", choices=["local", "openai"])
    parser.add_argument("--judge-model", default="models/Qwen__Qwen2.5-VL-7B-Instruct", help="Local judge model path/id")
    parser.add_argument("--openai-model", default="qwen3-max")
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--out", required=True, help="Output judged report json")
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def load_judge_model_and_tokenizer(model_id: str, dtype_name: str):
    dtype = resolve_dtype(dtype_name)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return "text", tok, mdl
    except ValueError:
        # Fallback for multimodal chat models like Qwen2.5-VL.
        from transformers import (
            AutoTokenizer as _AutoTokenizer,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )

        image_processor = Qwen2VLImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        video_processor = Qwen2VLVideoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = _AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
        mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return "vl", processor, mdl


def parse_judge_json(text: str) -> tuple[int, str]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = json.loads(block)
            score = int(obj.get("score", 0))
            reason = str(obj.get("reason", "")).strip()
            score = min(3, max(0, score))
            return score, reason
        except json.JSONDecodeError:
            pass
    t = text.lower()
    for score in (3, 2, 1, 0):
        marker_json = f'"score": {score}'
        marker_plain = f"score: {score}"
        if marker_json in t or marker_plain in t or t.strip().startswith(str(score)):
            return score, "fallback parse"
    return 0, "fallback parse"


def judge_with_openai(
    api_key: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    base_url: str | None = None,
) -> str:
    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_new_tokens,
        messages=[
            {"role": "system", "content": "You are a precise evaluation assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    args = parse_args()
    pred_data = json.load(open(args.pred_json, "r", encoding="utf-8"))
    rows = pred_data["rows"]

    use_local = args.judge_backend == "local"
    if use_local:
        model_type, tok_or_proc, model = load_judge_model_and_tokenizer(args.judge_model, args.dtype)
        model.to(args.device)
        model.eval()
    else:
        api_key = (
            args.openai_api_key
            or os.environ.get("DASHSCOPE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        base_url = (
            args.openai_base_url
            or os.environ.get("DASHSCOPE_BASE_URL", "")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY or OPENAI_API_KEY is required for --judge-backend openai")

    judged_rows = []
    if use_local:
        with torch.no_grad():
            for r in rows:
                prompt = JUDGE_PROMPT.format(
                    question=r["question"],
                    reference=r["gt"],
                    prediction=r["pred"],
                )
                messages = [{"role": "user", "content": prompt}]
                if model_type == "text":
                    tokenizer = tok_or_proc
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([text], return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                else:
                    processor = tok_or_proc
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], return_tensors="pt", padding=True)
                    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
                gen = out[0, inputs["input_ids"].shape[1] :]
                if model_type == "text":
                    judge_text = tok_or_proc.decode(gen, skip_special_tokens=True).strip()
                else:
                    judge_text = tok_or_proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
                score, reason = parse_judge_json(judge_text)

                rr = dict(r)
                rr["judge_score"] = int(score)
                rr["judge_reason"] = reason
                rr["judge_raw"] = judge_text
                judged_rows.append(rr)
    else:
        for r in rows:
            prompt = JUDGE_PROMPT.format(
                question=r["question"],
                reference=r["gt"],
                prediction=r["pred"],
            )
            judge_text = judge_with_openai(
                api_key=api_key,
                model=args.openai_model,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                base_url=base_url,
            )
            score, reason = parse_judge_json(judge_text)

            rr = dict(r)
            rr["judge_score"] = int(score)
            rr["judge_reason"] = reason
            rr["judge_raw"] = judge_text
            judged_rows.append(rr)

    n = max(1, len(judged_rows))
    judged_score_mean = float(sum(x["judge_score"] for x in judged_rows) / n)
    judged_score_norm = float(judged_score_mean / 3.0)
    out_data = {
        "meta": {
            "pred_json": args.pred_json,
            "judge_backend": args.judge_backend,
            "judge_model": args.judge_model if use_local else args.openai_model,
            "official_style": "MMBench-Video paper prompt reproduction with 0-3 semantic scoring",
        },
        "summary": {
            "num_samples": len(judged_rows),
            "judge_score_mean": judged_score_mean,
            "judge_score_norm": judged_score_norm,
        },
        "rows": judged_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved judged report: {out_path}")


if __name__ == "__main__":
    main()
