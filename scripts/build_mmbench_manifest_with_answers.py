#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MMBench-Video manifest with question_id and answer")
    parser.add_argument("--questions", default="data/MMBench-Video-10G/MMBench-Video_q.json")
    parser.add_argument("--answers", default="data/MMBench-Video-10G/MMBench-Video_a.json")
    parser.add_argument("--video-dir", default="data/MMBench-Video-10G/video_small")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--out", required=True, help="Output JSONL manifest path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_rows = json.load(open(args.questions, "r", encoding="utf-8"))
    a_rows = json.load(open(args.answers, "r", encoding="utf-8"))
    answer_by_qid = {r["question_id"]: r["answer"] for r in a_rows}

    video_dir = Path(args.video_dir)
    out_rows = []
    for q in q_rows:
        video_name = q["video_name"]
        video_path = video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            continue
        qid = q["question_id"]
        ans = answer_by_qid.get(qid)
        if ans is None:
            continue
        out_rows.append(
            {
                "id": qid,
                "video": str(video_path.resolve()),
                "prompt": q["question"],
                "question_id": qid,
                "video_name": video_name,
                "answer": ans,
                "dimensions": q.get("dimensions", ""),
                "video_type": q.get("video_type", ""),
            }
        )
        if args.limit > 0 and len(out_rows) >= args.limit:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"saved manifest: {out_path} (rows={len(out_rows)})")


if __name__ == "__main__":
    main()

