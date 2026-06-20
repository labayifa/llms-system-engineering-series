# Lab 1 — Profile the Memory Hierarchy

**Book 2 · Part I (GPU architecture, memory hierarchy)**

---

## Hypothesis

A decode-heavy workload is memory-bound: the GPU's HBM (High-Bandwidth Memory) is near its bandwidth ceiling while the compute units (tensor cores) are mostly idle. This places the workload below the roofline diagonal.

## Background

The roofline model plots achieved performance (FLOP/s) against arithmetic intensity (FLOPs per byte moved). A workload above the diagonal is compute-bound; below it is memory-bound. Decode falls below: each step does a small number of multiplications (one token's forward pass) but must stream the full weight matrices and KV cache from HBM. NVIDIA Nsight Compute (`ncu`) reports both achieved compute throughput and DRAM bandwidth, letting you read your workload's position on the roofline.

## Setup

```bash
# CUDA toolkit must be installed (nsys and ncu ship with it)
pip install torch vllm

# Confirm ncu is available
ncu --version
```

## Procedure

1. Write a small script (`generate.py`) that generates a single 500-token completion using vLLM or a plain HuggingFace model, instrumented with a CUDA event or `torch.cuda.synchronize()` around the decode loop.
2. Profile with Nsight Compute:
   ```bash
   ncu --set full -o decode_profile python generate.py
   ```
3. Open the report (`ncu-ui decode_profile.ncu-rep`) and read:
   - **Achieved SM occupancy** — how busy are the tensor cores?
   - **DRAM throughput** — how close is HBM to its peak bandwidth?
   - **Arithmetic intensity** — compute / bytes moved.
4. Locate the workload on the H100/A100 roofline chart.
5. Repeat with a prefill-only run (no decode, just `input_ids` of 1000 tokens) and compare the two profiles.

## Expected Observations

- **Decode profile:** DRAM throughput near the HBM ceiling (1.5–2 TB/s for an H100); SM occupancy low (below 20%).
- **Prefill profile:** much higher SM occupancy (the parallel attention over many tokens is compute-intensive); still somewhat DRAM-limited for short sequences.
- The decode workload sits firmly below the roofline diagonal — confirming it is memory-bound, not compute-bound.
- The gap between tensor-core utilization and HBM utilization in decode is the visual proof that adding more compute (a faster GPU clock) would not help — bandwidth is the bottleneck.

## Why This Matters

This lab makes the memory-bandwidth bottleneck tangible rather than theoretical. Once you have seen a decode workload pinned to the memory wall on a roofline plot, the motivation for every technique in Part II becomes obvious: KV compression, GQA, quantization, and prefill–decode disaggregation all exist to reduce the number of bytes the GPU must stream per token. No algorithmic change will help until the root constraint is understood.

## Variant Experiments

- Profile a batched decode run (batch size 8 vs batch size 1) and observe how HBM throughput improves — batching amortizes the weight-load cost across multiple tokens per step.
- Profile FlashAttention vs a naive attention implementation during prefill; observe the DRAM bytes-moved difference.
- Use `nsys` instead of `ncu` for a system-level trace: see CPU-to-GPU copy time, CUDA kernel launches, and cudaMemcpy events in the timeline.
