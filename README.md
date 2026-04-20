# KVBench (Model-Free)

面向离线长视频 QA 的 KV Cache 压缩基线框架。

特点：
- 不依赖本地模型权重下载。
- 通过离线 `trace`（或合成数据）评估压缩策略。
- 内置基线：`H2O`、`SnapKV`、`PyramidKV`、`VLCache`、`StreamingCache`。
- 预置你的方案原型：`dynamic_freq_window`（动态滑窗 + 频域分析 + 特殊记忆保留）。

## 目录

- `src/kvbench/types.py`: KV 状态与上下文定义
- `src/kvbench/strategies/`: 各策略实现
- `src/kvbench/registry.py`: 策略注册与构造
- `src/kvbench/eval/offline.py`: 离线指标评估
- `src/kvbench/eval/trace.py`: 合成 trace + NPZ trace 加载
- `src/kvbench/eval/cli.py`: 命令行入口
- `configs/baselines/*.json`: 基线示例配置

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

使用合成 trace 评估：

```bash
kvbench-offline --strategy h2o --target-tokens 512
kvbench-offline --strategy snap --target-tokens 512
kvbench-offline --strategy pyramid --target-tokens 512
kvbench-offline --strategy vlcache --target-tokens 512
kvbench-offline --strategy streamingcache --target-tokens 512
kvbench-offline --strategy dynamic_freq_window --target-tokens 512
```

传参示例：

```bash
kvbench-offline --strategy h2o --target-tokens 384 --param ema_decay=0.95
kvbench-offline --strategy streamingcache --target-tokens 512 --param window_size=768 --param anchor_interval=48
kvbench-offline --strategy dynamic_freq_window --target-tokens 512 --param basekv_ratio=0.5 --param outlier_z=1.8
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

运行：

```bash
kvbench-offline --strategy vlcache --target-tokens 512 --trace /path/to/your_trace.npz
```

## 指标

当前提供：
- `avg_kept_tokens`: 平均保留 token 数
- `avg_compression_ratio`: 平均保留比例
- `important_recall`: 关键 token 召回率（由 `--important` 指定 token id 集合）

## 说明

- 这里的 baseline 是工程可复用骨架，方便你后续对接真实 VLM 推理链路。
- 等你有真实 attention/KV trace 后，可直接替换合成 trace 做对比实验。
