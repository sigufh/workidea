# KVBench (Model-Free + Real VLM Trace)

面向离线长视频 QA 的 KV Cache 压缩基线框架，支持：
- 不依赖权重下载的合成 trace 快速验证。
- 基于 HuggingFace 真实模型推理抽取 KV/attention trace。
- 统一离线评测与论文式对比表。

内置策略：
- `FullKV`
- `H2O`
- `SnapKV`
- `PyramidKV`
- `VLCache`
- `StreamingCache`
- `dynamic_freq_window`（频域引导 + Outlier-KV 感知 + 动态滑窗 + 特殊记忆保留）

## 目录

- `src/kvbench/types.py`: KV 状态与上下文定义
- `src/kvbench/strategies/`: 各策略实现
- `src/kvbench/registry.py`: 策略注册与构造
- `src/kvbench/eval/offline.py`: 离线指标评估
- `src/kvbench/eval/trace.py`: 合成 trace + NPZ trace 加载
- `src/kvbench/eval/cli.py`: 命令行入口
- `configs/baselines/*.json`: 基线示例配置

## 环境准备（Conda）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashwin
pip install -e .
```

真实模型 trace 抽取依赖：

```bash
pip install -e ".[real]"
```

下载真实实验模型（Qwen + LLaVA）：

```bash
python scripts/download_real_models.py \
  --models Qwen/Qwen2.5-VL-7B-Instruct llava-hf/LLaVA-NeXT-Video-7B-hf \
  --local-dir-root models
```

## 任务 1：README 基线任务（合成 trace）

运行所有策略：

```bash
kvbench-offline --strategy h2o --target-tokens 512
kvbench-offline --strategy fullkv --target-tokens 512
kvbench-offline --strategy snap --target-tokens 512
kvbench-offline --strategy pyramid --target-tokens 512
kvbench-offline --strategy vlcache --target-tokens 512
kvbench-offline --strategy streamingcache --target-tokens 512
kvbench-offline --strategy dynamic_freq_window --target-tokens 512
```

`dynamic_freq_window` 论文风格参数示例：

```bash
kvbench-offline \
  --strategy dynamic_freq_window \
  --target-tokens 512 \
  --param window_size=512 \
  --param min_window_size=192 \
  --param basekv_ratio=0.45 \
  --param outlier_budget_ratio=0.25 \
  --param low_freq_ratio=0.25 \
  --param outlier_z=1.8 \
  --param recency_boost=0.35 \
  --param outlier_min_keep=8
```

## 接入你自己的离线 trace

`load_trace_npz` 支持以下 NPZ 字段：
- `keys`: `[steps, layers, heads, tokens, dim]`
- `values`: `[steps, layers, heads, tokens, dim]`
- `attn`: `[steps, layers, heads, tokens]`
- `history`: `[steps, tokens, hist]`（可选，供频域策略使用）
- `modality`: `[tokens]`（0=vision, 1=text，可选）
- `sink_idx`: `[k]`（可选）
- `special_idx`: `[m]`（可选）
- `valid_lens`: `[steps]`（可选，真实生成中每步有效 token 数）
- `token_ids`: `[tokens]`（可选，真实 token id）

运行：

```bash
kvbench-offline --strategy vlcache --target-tokens 512 --trace /path/to/your_trace.npz
```

## 任务 2：真实长视频模型实验（Qwen + LLaVA）

推荐两条主线（均为学术界常见开源 VLM）：
- Qwen 系：`Qwen/Qwen2.5-VL-7B-Instruct`
- LLaMA 家族多模态系：`llava-hf/LLaVA-NeXT-Video-7B-hf`

### 2.1 抽取 Qwen 视频 trace

```bash
python scripts/extract_trace_vlm_hf.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --backend qwen25vl \
  --video /path/to/video.mp4 \
  --prompt "请总结该视频事件并回答问题：xxx" \
  --max-frames 96 \
  --sample-fps 1.0 \
  --max-new-tokens 64 \
  --out traces/qwen25vl_video_trace.npz \
  --vision-prefix-tokens 576
```

### 2.2 抽取 LLaVA-NeXT-Video trace

```bash
python scripts/extract_trace_vlm_hf.py \
  --model-id llava-hf/LLaVA-NeXT-Video-7B-hf \
  --backend llava_next_video \
  --video /path/to/video.mp4 \
  --prompt "请总结该视频事件并回答问题：xxx" \
  --max-frames 96 \
  --sample-fps 1.0 \
  --max-new-tokens 64 \
  --out traces/llava_next_video_trace.npz \
  --vision-prefix-tokens 576
```

### 2.3 运行论文式离线对比

```bash
python scripts/run_paper_benchmark.py \
  --traces traces/qwen25vl_video_trace.npz traces/llava_next_video_trace.npz \
  --target-tokens 1024 \
  --out reports/paper_benchmark_1024.json
```

### 2.4 四卡并行加速（推荐）

准备清单文件（见 `configs/real_manifest.example.jsonl`）：

```json
{"id":"video_001","video":"/abs/path/video_001.mp4","prompt":"..."}
{"id":"video_002","video":"/abs/path/video_002.mp4","prompt":"..."}
```

四卡并行抽 trace + 自动跑论文对比：

```bash
python scripts/run_real_experiment_4gpu.py \
  --manifest configs/real_manifest.example.jsonl \
  --gpus 0,1,2,3 \
  --max-frames 96 \
  --sample-fps 1.0 \
  --max-new-tokens 64 \
  --target-tokens 1024 \
  --out-dir runs/real_paper
```

输出：
- `runs/real_paper/traces/*.npz`: 每个视频 x 模型的 trace
- `runs/real_paper/traces/*.log`: 每个并行任务日志
- `runs/real_paper/reports/paper_benchmark_t1024.json`: 汇总对比结果

如果只看单策略：

```bash
kvbench-offline --strategy dynamic_freq_window --trace traces/qwen25vl_video_trace.npz --target-tokens 1024
kvbench-offline --strategy dynamic_freq_window --trace traces/llava_next_video_trace.npz --target-tokens 1024
```

## 指标

当前提供：
- `avg_kept_tokens`: 平均保留 token 数
- `avg_compression_ratio`: 平均保留比例
- `important_recall`: 关键 token 召回率（由 `--important` 指定 token id 集合）

## 实验协议建议

- 固定 `target_tokens`（如 `512/1024/2048`）并报告每个预算点结果。
- 对同一视频集使用统一提示词模板与统一 `max_new_tokens`。
- 在 Qwen 与 LLaVA 两条线分别做主结果表，再汇总平均值。
- 至少报告压缩率与关键 token 召回；可追加任务准确率（若你有标注答案）。
