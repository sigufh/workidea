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
  --target-tokens 512 1024 2048 \
  --important auto \
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
  --attn-impl eager \
  --strict-attn \
  --target-tokens 512 1024 2048 \
  --out-dir runs/real_paper
```

如果 `manifest` 自带 `answer` 字段，并且你还要自动跑真实任务评分，可直接加上：

```bash
python scripts/run_real_experiment_4gpu.py \
  --manifest configs/mmbench_video_small_with_answers.jsonl \
  --gpus 0,1,2,3 \
  --max-frames 8 \
  --sample-fps 1.0 \
  --max-new-tokens 48 \
  --target-tokens 512 1024 2048 \
  --run-answer-eval \
  --judge-backend openai \
  --judge-model qwen3-max \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out-dir runs/real_paper_qwen_mmbench_4gpu
```

此时主流程除离线 trace benchmark 外，还会为每个模型额外生成：
- `mmbench_video_real_eval_<model>.json`: 真实答案级预测结果
- `mmbench_video_real_eval_<model>_qwen3-max_judged.json`: Qwen3 `0-3` 语义评分结果

### 2.5 逐策略真实评分（你要的“按答案打分”）

这一步会对每个压缩策略真实生成答案，再用 `qwen3-max` 做官方风格 `0-3` 语义打分，最后输出策略排名（不是 `attn_recall` 代理分）。

先确保环境变量：

```bash
export DASHSCOPE_API_KEY="你的百炼Key"
```

然后运行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_strategy_real_eval.py \
  --manifest configs/mmbench_video_small_with_answers.jsonl \
  --model-id /data/xiaotian/workidea/models/Qwen__Qwen2.5-VL-7B-Instruct \
  --backend qwen25vl \
  --strategies fullkv h2o snap pyramid vlcache streamingcache dynamic_freq_window \
  --target-tokens 1024 \
  --max-frames 2 \
  --sample-fps 1.0 \
  --max-new-tokens 48 \
  --judge-backend openai \
  --judge-model qwen3-max \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out-dir runs/strategy_real_eval_t1024
```

输出文件：
- `runs/strategy_real_eval_t1024/strategy_real_eval_<strategy>_t1024.json`: 每个策略的真实预测结果
- `runs/strategy_real_eval_t1024/strategy_real_eval_<strategy>_t1024_judged.json`: 每个策略的 Qwen3 判分结果
- `runs/strategy_real_eval_t1024/strategy_real_eval_ranking_t1024.json`: 按 `judge_score_norm` 排序后的最终排名

多档预算一键跑（`256/512/1024`）：

```bash
source ~/.profile

BASE=runs/strategy_real_eval_multibudget_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BASE"
for T in 256 512 1024; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_strategy_real_eval.py \
    --manifest configs/mmbench_video_small_with_answers.jsonl \
    --model-id /data/xiaotian/workidea/models/Qwen__Qwen2.5-VL-7B-Instruct \
    --backend qwen25vl \
    --strategies fullkv h2o snap pyramid vlcache streamingcache dynamic_freq_window \
    --target-tokens "$T" \
    --max-frames 2 \
    --sample-fps 1.0 \
    --max-new-tokens 48 \
    --judge-backend openai \
    --judge-model qwen3-max \
    --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --out-dir "$BASE/t$T"
done
echo "done: $BASE"
```

多档结果汇总文件：
- `$BASE/t256/strategy_real_eval_ranking_t256.json`
- `$BASE/t512/strategy_real_eval_ranking_t512.json`
- `$BASE/t1024/strategy_real_eval_ranking_t1024.json`

你这台机器（本地 Qwen2.5-VL + 百炼 Qwen3 judge）建议先配置环境变量再运行：

```bash
export DASHSCOPE_API_KEY="你的百炼Key"
```

然后执行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_real_experiment_4gpu.py \
  --manifest configs/mmbench_video_small_with_answers.jsonl \
  --models /data/xiaotian/workidea/models/Qwen__Qwen2.5-VL-7B-Instruct \
  --gpus 0,1,2,3 \
  --max-frames 2 \
  --sample-fps 1.0 \
  --max-new-tokens 48 \
  --target-tokens 512 1024 2048 \
  --run-answer-eval \
  --judge-backend openai \
  --judge-model qwen3-max \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out-dir runs/real_paper_qwen_mmbench_4gpu_f2_clean
```

安全建议：不要把真实 `DASHSCOPE_API_KEY` 写入 README、脚本或提交到仓库。

如果你只想先验证报告生成链路（不重新抽 trace），可以单独跑：

```bash
python scripts/run_paper_benchmark.py \
  --traces runs/real_paper_qwen_mmbench_4gpu_f2_clean/traces/<trace1>.npz runs/real_paper_qwen_mmbench_4gpu_f2_clean/traces/<trace2>.npz \
  --target-tokens 512 1024 2048 \
  --important auto \
  --out runs/real_paper_qwen_mmbench_4gpu_f2_clean/reports/paper_benchmark_multi_budget_partial.json
```

输出：
- `runs/real_paper/traces/*.npz`: 每个视频 x 模型的 trace
- `runs/real_paper/traces/*.log`: 每个并行任务日志
- `runs/real_paper/reports/paper_benchmark_t1024.json`: 汇总对比结果
- `runs/real_paper/reports/paper_benchmark_multi_budget.json`: 多预算汇总结果
- `runs/real_paper/reports/paper_benchmark_multi_budget.md`: 论文风格排名表（按 budget 分组）

如果只看单策略：

```bash
kvbench-offline --strategy dynamic_freq_window --trace traces/qwen25vl_video_trace.npz --target-tokens 1024
kvbench-offline --strategy dynamic_freq_window --trace traces/llava_next_video_trace.npz --target-tokens 1024
```

## 指标

当前提供：
- `avg_kept_tokens`: 平均保留 token 数
- `std_kept_tokens`: 保留 token 数标准差（稳定性）
- `avg_compression_ratio`: 平均保留比例
- `std_compression_ratio`: 保留比例标准差
- `compression_ratio_p50/p90`: 保留比例分位点
- `important_recall`: 关键 token 召回率（由 `--important` 指定 token id 集合）
- `sink_recall`: sink token 召回率（若 trace 含 sink 标注）
- `special_recall`: special memory token 召回率（若 trace 含标注）
- `attention_mass_recall`: 注意力质量代理（保留 token 承载的注意力质量占比）
- `efficiency_score`: 质量-压缩调和分
- `quality_retention_pct`: 相对 FullKV 的质量保真率（学术评测主指标）
- `compression_factor`: 压缩倍数（`1 / avg_compression_ratio`）

学术评测协议（当前实现）：
- `FullKV` 作为基线单列展示，不参与压缩方法排名。
- 压缩方法按 `quality_retention_pct` 排名；`compression_ratio/factor` 与各类 recall 单独报告，不再使用自定义加权综合分。

注意：
- 若 `attn` 提取失败（全 0），`attention_mass_recall/efficiency_score` 会失效，建议抽 trace 时使用 `--attn-impl eager --strict-attn`。
- `important` 建议默认 `auto`，会优先使用 trace 中的 special/sink 标注按位置计算召回，避免固定 token id 与真实样本不匹配。
- `dynamic_freq_window` 已升级为 V2（phase-aware + layer-aware + unified scoring），结构说明见 `docs/dynamic_freq_window_v2.md`。

## 实验协议建议

- 固定 `target_tokens`（如 `512/1024/2048`）并报告每个预算点结果。
- 对同一视频集使用统一提示词模板与统一 `max_new_tokens`。
- 在 Qwen 与 LLaVA 两条线分别做主结果表，再汇总平均值。
- 至少报告压缩率与关键 token 召回；可追加任务准确率（若你有标注答案）。

## 3. 其他设备拉取后复现（完整流程）

### 3.1 拉取代码

```bash
git clone https://github.com/sigufh/workidea.git
cd workidea
```

### 3.2 环境安装

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n flashwin python=3.10 -y
conda activate flashwin
pip install -e .
pip install -e ".[real]"
```

### 3.3 配置云端判分 Key（百炼）

```bash
export DASHSCOPE_API_KEY="你的百炼Key"
```

建议写入 `~/.profile` 以便新终端生效。

### 3.4 下载模型

```bash
python scripts/download_real_models.py \
  --models Qwen/Qwen2.5-VL-7B-Instruct llava-hf/LLaVA-NeXT-Video-7B-hf \
  --local-dir-root models
```

### 3.5 快速冒烟（小样本）

```bash
python scripts/build_mmbench_manifest_with_answers.py \
  --questions data/MMBench-Video-10G/MMBench-Video_q.json \
  --answers data/MMBench-Video-10G/MMBench-Video_a.json \
  --video-dir data/MMBench-Video-10G/video_small \
  --limit 16 \
  --out configs/mmbench_video_smoke16_with_answers.jsonl

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_real_experiment_4gpu.py \
  --manifest configs/mmbench_video_smoke16_with_answers.jsonl \
  --models models/Qwen__Qwen2.5-VL-7B-Instruct \
  --gpus 0,1,2,3 \
  --max-frames 2 \
  --sample-fps 1.0 \
  --max-new-tokens 48 \
  --target-tokens 256 512 1024 \
  --run-answer-eval \
  --judge-backend openai \
  --judge-model qwen3-max \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out-dir runs/real_smoke16_qwen
```

## 4. 完整数据集真实论文实验（大盘复现）

### 4.0 全量数据集下载与目录准备

MMBench-Video-10G 需要你先拿到官方发布的数据文件（`MMBench-Video_q.json`、`MMBench-Video_a.json`、视频文件）。

推荐目录结构：

```text
data/MMBench-Video-10G/
  MMBench-Video_q.json
  MMBench-Video_a.json
  video_small/            # 或 video/（完整视频目录）
    *.mp4
```

如果你拿到的是压缩包，在仓库根目录执行（按你的实际文件名替换）：

```bash
mkdir -p data/MMBench-Video-10G
tar -xvf MMBench-Video-10G_video_small.tar -C data/MMBench-Video-10G/
tar -xvf MMBench-Video-10G_annotations.tar -C data/MMBench-Video-10G/
```

快速校验：

```bash
python - <<'PY'
import json, os, glob
root = "data/MMBench-Video-10G"
q = os.path.join(root, "MMBench-Video_q.json")
a = os.path.join(root, "MMBench-Video_a.json")
vsmall = os.path.join(root, "video_small")
vfull = os.path.join(root, "video")
print("q_exists:", os.path.exists(q))
print("a_exists:", os.path.exists(a))
if os.path.exists(vsmall):
    print("video_small_mp4:", len(glob.glob(os.path.join(vsmall, "*.mp4"))))
if os.path.exists(vfull):
    print("video_mp4:", len(glob.glob(os.path.join(vfull, "*.mp4"))))
if os.path.exists(q):
    print("questions:", len(json.load(open(q, "r", encoding="utf-8"))))
if os.path.exists(a):
    print("answers:", len(json.load(open(a, "r", encoding="utf-8"))))
PY
```

### 4.1 构建全量 manifest（不设上限）

如果你有更大磁盘，建议直接全量：

```bash
python scripts/build_mmbench_manifest_with_answers.py \
  --questions data/MMBench-Video-10G/MMBench-Video_q.json \
  --answers data/MMBench-Video-10G/MMBench-Video_a.json \
  --video-dir data/MMBench-Video-10G/video_small \
  --limit 0 \
  --out configs/mmbench_video_full_with_answers.jsonl
```

如果你存的是完整视频目录而不是 `video_small`，把 `--video-dir` 改成实际路径（例如 `data/MMBench-Video-10G/video`）。

### 4.2 跑完整论文主流程（多预算）

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_real_experiment_4gpu.py \
  --manifest configs/mmbench_video_full_with_answers.jsonl \
  --models models/Qwen__Qwen2.5-VL-7B-Instruct models/llava-hf__LLaVA-NeXT-Video-7B-hf \
  --gpus 0,1,2,3 \
  --max-frames 8 \
  --sample-fps 1.0 \
  --max-new-tokens 64 \
  --attn-impl eager \
  --strict-attn \
  --target-tokens 256 512 1024 2048 \
  --run-answer-eval \
  --judge-backend openai \
  --judge-model qwen3-max \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out-dir runs/real_paper_mmbench_full
```

### 4.3 跑“逐策略真实答案评分”（可单模型）

```bash
for T in 256 512 1024 2048; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_strategy_real_eval.py \
    --manifest configs/mmbench_video_full_with_answers.jsonl \
    --model-id models/Qwen__Qwen2.5-VL-7B-Instruct \
    --backend qwen25vl \
    --strategies fullkv h2o snap pyramid vlcache streamingcache dynamic_freq_window \
    --target-tokens "$T" \
    --max-frames 8 \
    --sample-fps 1.0 \
    --max-new-tokens 64 \
    --judge-backend openai \
    --judge-model qwen3-max \
    --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --out-dir runs/strategy_real_eval_full/t$T
done
```

关键输出：
- `runs/real_paper_mmbench_full/reports/paper_benchmark_multi_budget.json`
- `runs/real_paper_mmbench_full/reports/paper_benchmark_multi_budget.md`
- `runs/strategy_real_eval_full/t*/strategy_real_eval_ranking_t*.json`
