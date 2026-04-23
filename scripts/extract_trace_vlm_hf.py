#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract KV/attention trace from HF video-language models")
    parser.add_argument("--model-id", required=True, help="HF model id")
    parser.add_argument("--video", required=True, help="Local video path")
    parser.add_argument("--prompt", required=True, help="Question/instruction about the video")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen25vl", "llava_next_video"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention implementation. Use eager when you need non-empty output_attentions.",
    )
    parser.add_argument(
        "--strict-attn",
        action="store_true",
        help="Fail if extracted attention is all zeros.",
    )
    parser.add_argument(
        "--vision-prefix-tokens",
        type=int,
        default=0,
        help="Mark first N tokens as vision. If unknown, keep 0 and rely on sink/special marks.",
    )
    parser.add_argument("--sink-count", type=int, default=8)
    parser.add_argument("--special-idx", default="", help="Comma-separated token positions to preserve")
    parser.add_argument(
        "--save-compressed",
        action="store_true",
        help="Save trace with np.savez_compressed (smaller but slower). Default uses np.savez for faster completion.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def detect_backend(backend: str, model_id: str) -> str:
    if backend != "auto":
        return backend
    low = model_id.lower()
    if "qwen2.5-vl" in low or "qwen-vl" in low:
        return "qwen25vl"
    if "llava" in low and "video" in low:
        return "llava_next_video"
    raise ValueError(
        "Cannot auto-detect backend. Please pass --backend qwen25vl or --backend llava_next_video."
    )


def sample_video_frames(path: str, sample_fps: float, max_frames: int) -> list[np.ndarray]:
    vr = VideoReader(path, ctx=cpu(0))
    if len(vr) == 0:
        raise ValueError(f"Video has no frames: {path}")

    native_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 25.0
    stride = max(1, int(round(native_fps / max(sample_fps, 1e-3))))
    frame_idx = list(range(0, len(vr), stride))
    if len(frame_idx) > max_frames:
        lin = np.linspace(0, len(frame_idx) - 1, num=max_frames, dtype=np.int64)
        frame_idx = [frame_idx[int(i)] for i in lin]

    batch = vr.get_batch(frame_idx).asnumpy()  # [T, H, W, C], uint8
    return [frame for frame in batch]


def build_inputs(
    processor,
    backend: str,
    video_path: str,
    prompt: str,
    frames: list[np.ndarray],
):
    if backend not in {"qwen25vl", "llava_next_video"}:
        raise ValueError(f"Unsupported backend: {backend}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        # Most recent processors accept list-of-frames in "videos".
        inputs = processor(
            text=[text],
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )
    except TypeError:
        # Fallback for variants expecting non-nested frames.
        inputs = processor(
            text=[text],
            videos=frames,
            return_tensors="pt",
            padding=True,
        )
    return inputs


def step_attention_to_vector(attn_tuple) -> np.ndarray:
    if attn_tuple is None or len(attn_tuple) == 0 or attn_tuple[0] is None:
        return np.zeros((1,), dtype=np.float32)
    a0 = attn_tuple[0]
    if a0.ndim == 4:
        vec = a0[0, :, -1, :].mean(dim=0).detach().float().cpu().numpy()
    elif a0.ndim == 3:
        vec = a0[:, -1, :].mean(dim=0).detach().float().cpu().numpy()
    elif a0.ndim == 2:
        vec = a0.mean(dim=0).detach().float().cpu().numpy()
    else:
        return np.zeros((1,), dtype=np.float32)
    return vec.astype(np.float32)


def to_head_token_attention(attn_tensor, heads: int, tokens: int) -> np.ndarray:
    if attn_tensor is None:
        return np.zeros((heads, tokens), dtype=np.float32)

    t = attn_tensor.detach().float().cpu()
    if t.ndim == 4:
        m = t[0, :, -1, :]
    elif t.ndim == 3:
        m = t[:, -1, :]
    elif t.ndim == 2:
        m = t
    else:
        return np.zeros((heads, tokens), dtype=np.float32)

    arr = m.numpy().astype(np.float32)
    if arr.shape[0] != heads:
        if arr.shape[0] > heads:
            arr = arr[:heads]
        else:
            pad_h = np.zeros((heads - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad_h], axis=0)
    if arr.shape[1] != tokens:
        if arr.shape[1] > tokens:
            arr = arr[:, :tokens]
        else:
            pad_t = np.zeros((arr.shape[0], tokens - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad_t], axis=1)
    return arr


def _model_load_kwargs(dtype, attn_impl: str) -> dict:
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if attn_impl != "auto":
        kwargs["attn_implementation"] = attn_impl
    return kwargs


def load_processor_and_model(model_id: str, backend: str, dtype_name: str, attn_impl: str):
    dtype = resolve_dtype(dtype_name)
    load_kwargs = _model_load_kwargs(dtype=dtype, attn_impl=attn_impl)
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
            **load_kwargs,
        )
        model.config.output_attentions = True
        return processor, model

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model.config.output_attentions = True
    return processor, model


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(out_path.suffix + ".progress.json")

    def write_progress(step: int, total_steps: int, phase: str) -> None:
        tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
        payload = {
            "step": int(step),
            "total_steps": int(total_steps),
            "phase": phase,
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, progress_path)

    backend = detect_backend(args.backend, args.model_id)
    frames = sample_video_frames(args.video, args.sample_fps, args.max_frames)

    processor, model = load_processor_and_model(args.model_id, backend, args.dtype, args.attn_impl)
    model.to(args.device)
    model.eval()

    inputs = build_inputs(
        processor=processor,
        backend=backend,
        video_path=args.video,
        prompt=args.prompt,
        frames=frames,
    )
    model_inputs = {}
    for k, v in inputs.items():
        model_inputs[k] = v.to(model.device) if hasattr(v, "to") else v

    steps_keys: list[list[np.ndarray]] = []
    steps_values: list[list[np.ndarray]] = []
    steps_attn: list[list[np.ndarray]] = []
    attn_snapshots: list[np.ndarray] = []
    valid_lens: list[int] = []

    with torch.no_grad():
        total_steps = max(1, args.max_new_tokens)
        write_progress(step=0, total_steps=total_steps, phase="prefill")
        outputs = model(use_cache=True, output_attentions=True, **model_inputs)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        token_ids = model_inputs["input_ids"][0].tolist() + [int(next_token.item())]

        k_layers = []
        v_layers = []
        a_layers = []
        for l, kv in enumerate(past):
            k = kv[0][0].detach().float().cpu().numpy()
            v = kv[1][0].detach().float().cpu().numpy()
            k_layers.append(k.astype(np.float32))
            v_layers.append(v.astype(np.float32))
            attn_l = outputs.attentions[l] if outputs.attentions is not None and l < len(outputs.attentions) else None
            a_layers.append(to_head_token_attention(attn_l, k.shape[0], k.shape[1]))
        steps_keys.append(k_layers)
        steps_values.append(v_layers)
        steps_attn.append(a_layers)
        attn_snapshots.append(a_layers[0].mean(axis=0).astype(np.float32) if a_layers else np.zeros((1,), dtype=np.float32))
        valid_lens.append(k_layers[0].shape[1])
        write_progress(step=1, total_steps=total_steps, phase="decode")

        for _ in range(max(0, args.max_new_tokens - 1)):
            outputs = model(
                input_ids=next_token.to(model.device),
                use_cache=True,
                past_key_values=past,
                output_attentions=True,
            )
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_ids.append(int(next_token.item()))

            k_layers = []
            v_layers = []
            a_layers = []
            for l, kv in enumerate(past):
                k = kv[0][0].detach().float().cpu().numpy()
                v = kv[1][0].detach().float().cpu().numpy()
                k_layers.append(k.astype(np.float32))
                v_layers.append(v.astype(np.float32))
                attn_l = outputs.attentions[l] if outputs.attentions is not None and l < len(outputs.attentions) else None
                a_layers.append(to_head_token_attention(attn_l, k.shape[0], k.shape[1]))
            steps_keys.append(k_layers)
            steps_values.append(v_layers)
            steps_attn.append(a_layers)
            attn_snapshots.append(a_layers[0].mean(axis=0).astype(np.float32) if a_layers else np.zeros((1,), dtype=np.float32))
            valid_lens.append(k_layers[0].shape[1])
            write_progress(step=len(valid_lens), total_steps=total_steps, phase="decode")

    steps = len(steps_keys)
    layers = len(steps_keys[0])
    heads = steps_keys[0][0].shape[0]
    dim = steps_keys[0][0].shape[2]
    max_tokens = max(valid_lens)

    keys = np.zeros((steps, layers, heads, max_tokens, dim), dtype=np.float32)
    values = np.zeros((steps, layers, heads, max_tokens, dim), dtype=np.float32)
    attn = np.zeros((steps, layers, heads, max_tokens), dtype=np.float32)

    for s in range(steps):
        cur = valid_lens[s]
        for l in range(layers):
            keys[s, l, :, :cur, :] = steps_keys[s][l]
            values[s, l, :, :cur, :] = steps_values[s][l]
            attn[s, l, :, :cur] = steps_attn[s][l]

    attn_nonzero = bool(np.any(attn != 0))
    if not attn_nonzero:
        msg = (
            "warning: extracted attn is all zeros. "
            "Try --attn-impl eager and avoid flash-attn kernels when collecting output_attentions."
        )
        if args.strict_attn:
            raise RuntimeError(msg)
        print(msg)

    history = np.zeros((steps, max_tokens, steps), dtype=np.float32)
    for s in range(steps):
        for t in range(s + 1):
            vec = attn_snapshots[t]
            history[s, : vec.shape[0], t] = vec

    modality = np.ones((max_tokens,), dtype=np.int64)  # default text
    if args.vision_prefix_tokens > 0:
        modality[: min(args.vision_prefix_tokens, max_tokens)] = 0

    sink_idx = np.arange(min(args.sink_count, max_tokens), dtype=np.int64)
    special_idx = (
        np.array([int(x) for x in args.special_idx.split(",") if x.strip()], dtype=np.int64)
        if args.special_idx.strip()
        else np.array([], dtype=np.int64)
    )

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    write_progress(step=total_steps, total_steps=total_steps, phase="finalize")
    saver = np.savez_compressed if args.save_compressed else np.savez
    write_progress(step=total_steps, total_steps=total_steps, phase="save")
    # Use file object to prevent numpy from auto-appending ".npz" to tmp path.
    with open(tmp_out, "wb") as f:
        saver(
            f,
            keys=keys,
            values=values,
            attn=attn,
            history=history,
            modality=modality,
            sink_idx=sink_idx,
            special_idx=special_idx,
            valid_lens=np.array(valid_lens, dtype=np.int64),
            token_ids=np.array(token_ids, dtype=np.int64),
            model_id=np.array([args.model_id]),
            backend=np.array([backend]),
            video_path=np.array([args.video]),
            attn_nonzero=np.array([1 if attn_nonzero else 0], dtype=np.int64),
            attn_impl=np.array([args.attn_impl]),
        )
    os.replace(tmp_out, out_path)
    try:
        progress_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"saved trace: {out_path}")


if __name__ == "__main__":
    main()
