# Lab 2 — Watch KV Cache Growth

**Book 1 · Part II (memory bandwidth, KV cache formula)**

---

## Hypothesis

VRAM usage climbs linearly with sequence length during a long generation, and decode throughput (tokens/sec) drifts downward as the cache grows — because each decode step must read a larger KV tensor from HBM.

## Background

For every new token generated, the model appends one row to the K and V matrices for every layer. The KV cache size at step t is:

```
bytes = 2 × n_layers × n_kv_heads × d_head × t × bytes_per_element
```

The `2` covers K and V; `t` is the current sequence length. For a Llama 3.1 70B model with a 32k context, this reaches 80 GB for a single request — before weights. Decode is memory-bandwidth-bound: each step reads the whole accumulated cache from HBM, so throughput degrades as that cache grows.

## Setup

```bash
pip install vllm nvidia-ml-py matplotlib

# Terminal 1 — start the model server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# Terminal 2 — poll GPU memory every second
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
    | tee vram_log.csv
```

## Procedure

1. Generate a single long completion (~4000 output tokens) from a short prompt.
2. In a parallel terminal, sample VRAM (MiB used) every second with `nvidia-smi`.
3. Inside the generation loop, record wall-clock time at every 100-token boundary to compute tokens/sec in sliding windows.
4. After generation, plot two figures on the same time axis:
   - VRAM used (MiB) vs time → should be a straight line.
   - Tokens/sec (sliding window) vs time → should trend downward.
5. Use the KV cache formula above to predict the slope of the VRAM curve; compare to the observed slope.

## Expected Observations

- **VRAM rises in a straight line** — linear in sequence length, confirming the formula.
- **Tokens/sec drifts downward** as the sequence lengthens — because each step reads a larger KV tensor from HBM.
- The predicted slope from the formula should match the observed slope within measurement noise.
- The throughput drop is gradual, not abrupt — this is a soft degradation, not a cliff.

## Why This Matters

This lab makes the KV cache formula concrete. Once you can predict the slope of the VRAM curve from model hyperparameters, you can size GPU memory for a given context length before deployment, choose between MHA and GQA models based on your context requirements, and understand why quantizing the KV cache (to INT8 or FP8) is a meaningful optimization even if the weights are already quantized.

## Variant Experiments

- Repeat with a GQA model (e.g., Llama 3.1 8B, which uses 8 KV heads vs 32 query heads) and compare the VRAM slope to a full-MHA model — the 4× reduction should be visible in the slope.
- Enable `--kv-cache-dtype fp8` in vLLM and compare the VRAM slope to the default FP16 cache.
- Run the same generation at batch size 1 vs batch size 4 and confirm VRAM scales linearly with concurrent sequences.
