# Dynamic Freq Window V2

This document records the structure of the optimized `dynamic_freq_window` strategy and the rationale behind its design.

## Objective

At fixed KV budget, improve quality retention by reducing wrong-token selection under multimodal traces.

## Core Changes

Compared with the earlier version, V2 introduces:

1. Unified importance scoring:
- Merge current attention, EMA attention, attention-history mean, low-frequency stability, and recency into one score.

2. Phase-aware weighting:
- Use decode phase ratio (`current_step / token_count`) to shift focus:
  - Early phase: more history/low-frequency stability.
  - Late phase: more current attention/recency.

3. Layer-aware weighting:
- Use layer depth ratio (`layer_idx / (num_layers-1)`) to shift focus:
  - Shallow layers: slightly more global/history terms.
  - Deep layers: slightly more current attention/recency terms.

4. Soft recency and modality floor:
- Recency is no longer hard-reserved as a large window.
- Keep minimum text and vision quotas to avoid modality collapse.

5. Conservative outlier budget:
- Outlier quota is reduced and acts as a soft bonus, not dominant selection.

## Selection Pipeline

For each layer:

1. Build feature signals:
- `cur_attn_n`: normalized current attention
- `ema_attn_n`: normalized EMA attention (fallback to `cur_attn_n`)
- `hist_mean`: normalized mean attention history
- `base_score`: normalized low-frequency stability from FFT
- `recency_vec`: normalized recency ramp in adaptive recent window

2. Compute phase/layer adaptive weights:
- Inputs:
  - `phase_ratio in [0, 1]`
  - `layer_ratio in [0, 1]`
- Output normalized tuple:
  - `(cur_w, ema_w, hist_w, lowf_w, rec_w)`

3. Compute unified score:
- `mixed_base = cur_w*cur_attn_n + ema_w*ema_attn_n + hist_w*hist_mean + lowf_w*base_score + rec_w*recency_vec`

4. Candidate assembly:
- `special_pick` (sink/special memory)
- `recent_pick` (small quota)
- `base_pick` (top-k by unified score)
- `outlier_pick` (high-frequency/spike based, bounded quota)
- `text_pick` and `vision_pick` (modality floor)

5. Budget reconciliation:
- Merge all picks.
- If under budget: fill by top-k `mixed_base`.
- If over budget: preserve special first, then top score among remaining.

## Key Parameters (Current Defaults)

- Window:
  - `window_size=384`
  - `min_window_size=64`
- Budget split:
  - `basekv_ratio=0.7`
  - `outlier_budget_ratio=0.1`
  - `recency_min_ratio=0.08`
- Outlier:
  - `low_freq_ratio=0.2`
  - `outlier_z=2.2`
  - `outlier_min_keep=8`
- Modality floor:
  - `text_min_ratio=0.15`
  - `vision_min_ratio=0.15`
- Base weights:
  - `cur_attn_weight=0.55`
  - `ema_weight=0.2`
  - `history_weight=0.15`
  - `lowfreq_weight=0.1`
  - `recency_boost=0.2`

## Practical Notes

- V2 keeps the same strategy name (`dynamic_freq_window`) for compatibility.
- For strict A/B comparison, run benchmark on the same trace set before/after code change.
- If you want frozen reproducibility, pin this file version plus strategy source hash in experiment metadata.
