# Lab 4 — Build a Cost Model

**Book 2 · Part V (cost modeling)**

---

## Hypothesis

Cost per million tokens falls steeply as batch size increases from 1 toward the memory or latency limit, then flattens — producing a characteristic elbow curve. This makes batch size the single most important serving economics lever.

## Background

The fundamental cost identity:

```
cost_per_token = gpu_hourly_rate / (3600 × tokens_per_second)
cost_per_request = cost_per_token × (input_tokens + output_tokens)
```

At batch size 1, decode is severely memory-bound: the GPU streams weights and KV cache from HBM for a single token per step, leaving tensor cores mostly idle. As batch size increases, each forward pass generates one token *per sequence in the batch* — the weight-load cost is amortized across N tokens instead of 1. Cost per token falls roughly as `1/N` until memory fills (more sequences → more KV cache → VRAM exhausted) or latency SLAs are violated (larger batches → longer queue waits).

## Setup

```bash
pip install vllm matplotlib

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

You don't need a real billing account. Use the GPU's spot instance price (or an estimated on-premises amortized cost) as `gpu_hourly_rate`.

## Procedure

1. Write a function that sends N concurrent requests to the vLLM server (all with the same prompt and `max_tokens=200`) and records elapsed wall-clock time from first request to last response.
2. Compute tokens/sec for each N: `(N × 200) / elapsed`.
3. Sweep batch sizes: N ∈ {1, 2, 4, 8, 16, 32, 64} (stop when VRAM OOMs or latency exceeds your budget).
4. Compute `cost_per_million_tokens` for each N using your GPU's hourly rate.
5. Plot cost/million tokens vs batch size (the elbow curve).
6. Annotate the point where marginal gain drops below 10% (the practical diminishing-returns threshold).

## Expected Observations

- **Cost drops sharply from batch 1 to batch 8**: often an order-of-magnitude reduction.
- **Diminishing returns set in past batch 16–32** as the GPU approaches full HBM bandwidth utilization.
- **The curve flattens then may rise** if batch size causes memory pressure (OOM or swap) or latency violations.
- The elbow point is your practical batch-size target for cost-optimized serving under a given latency SLA.

## Why This Matters

This lab is the connecting thread between the hardware chapters (memory-bound decode) and the operational chapters (serving economics). Every optimization discussed in Parts I–IV — quantization, GQA, prefix caching, speculative decoding — shows up in this cost model as either higher tokens/sec at the same batch size, or the ability to run a larger batch before hitting the memory ceiling. Understanding the cost curve also prevents over-engineering: if the elbow is at batch 16, investing in distributed serving for a workload that peaks at batch 8 is waste.

## Variant Experiments

- Repeat the sweep with INT8 quantization enabled (`--quantization awq`) and plot both curves on the same axes — observe the tokens/sec improvement and its cost effect.
- Add a latency-budget constraint: mark batch sizes where p99 TTFT exceeds 2 seconds on the same plot. The cost-optimal point is the highest batch size that still meets SLA.
- Compare a small model (3B) vs a medium model (7B) at the same batch size and GPU — show how model size shifts the elbow point.
