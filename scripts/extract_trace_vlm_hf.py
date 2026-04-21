#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        "--vision-prefix-tokens",
        type=int,
        default=0,
        help="Mark first N tokens as vision. If unknown, keep 0 and rely on sink/special marks.",
    )
    parser.add_argument("--sink-count", type=int, default=8)
    parser.add_argument("--special-idx", default="", help="Comma-separated token positions to preserve")
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
    a0 = attn_tuple[0][0]  # [heads, q, k]
    vec = a0[:, -1, :].mean(dim=0).detach().float().cpu().numpy()
    return vec.astype(np.float32)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = detect_backend(args.backend, args.model_id)
    frames = sample_video_frames(args.video, args.sample_fps, args.max_frames)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=True,
    )
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
            a_layers.append(outputs.attentions[l][0][:, -1, :].detach().float().cpu().numpy().astype(np.float32))
        steps_keys.append(k_layers)
        steps_values.append(v_layers)
        steps_attn.append(a_layers)
        attn_snapshots.append(step_attention_to_vector(outputs.attentions))
        valid_lens.append(k_layers[0].shape[1])

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
                a_layers.append(outputs.attentions[l][0][:, -1, :].detach().float().cpu().numpy().astype(np.float32))
            steps_keys.append(k_layers)
            steps_values.append(v_layers)
            steps_attn.append(a_layers)
            attn_snapshots.append(step_attention_to_vector(outputs.attentions))
            valid_lens.append(k_layers[0].shape[1])

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

    np.savez_compressed(
        out_path,
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
    )
    print(f"saved trace: {out_path}")


if __name__ == "__main__":
    main()
