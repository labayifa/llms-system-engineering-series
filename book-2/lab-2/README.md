# Lab 2 — Tiled vs Naive Matmul

**Book 2 · Part IV (kernels, tiling, arithmetic intensity)**

---

## Hypothesis

A tiled matrix-multiplication kernel beats a naive one despite performing the identical number of multiply-accumulate operations — the speedup comes entirely from reducing HBM traffic by reusing data in SRAM.

## Background

The naive matmul kernel loads each element of A and B from HBM for every output element it participates in. For an N×N multiply, each of the N² elements of A is loaded N times — O(N³) bytes moved, vs O(N²) bytes that actually need to exist. Tiling fixes this: load a tile of A and a tile of B into SRAM (shared memory), compute all the output contributions of that tile pair, then discard and load the next tile. Each element is loaded from HBM once per tile pass — O(N²·T) bytes instead of O(N³), where T is the tile size. Arithmetic intensity rises from O(1) to O(T), moving the kernel up the roofline toward the compute-bound region.

## Setup

```bash
# Requires an NVIDIA GPU with CUDA toolkit installed
pip install torch

# Optional: Triton for a Python-level tiled kernel
pip install triton
```

## Procedure

1. Implement a **naive CUDA matmul kernel** in CUDA C or Triton: one thread per output element, loading A and B directly from global memory on every access.
2. Implement a **tiled CUDA matmul kernel** using shared memory tiles of size T×T (e.g., T=16 or T=32). Load one tile of A and one tile of B into `__shared__` memory per block, sync threads, compute the tile dot-product, repeat for the next tile pair.
3. Benchmark both kernels on the same N×N matrices (N=512, 1024, 2048, 4096).
4. Profile HBM bytes moved for each:
   ```bash
   ncu --metrics dram__bytes.sum naive_kernel tiled_kernel
   ```
5. Plot wall-clock time and HBM bytes moved vs N for both kernels on the same axes.

## Expected Observations

- **The tiled kernel is several times faster** (commonly 4–10×) for large N.
- **HBM bytes moved by the tiled kernel is dramatically lower** despite computing the same result — this is the entire source of the speedup.
- FLOP counts are identical for both kernels — confirmed by the profiler's `sm__ops_...` metrics.
- Performance of the tiled kernel continues to improve as tile size T grows (up to the SRAM capacity of the SM).

## Why This Matters

This lab is the kernel-level explanation for why FlashAttention works. FlashAttention is a tiled attention kernel: it keeps the query, key, and value tiles in SRAM, computes the attention output tile-by-tile using an online softmax, and never materializes the full N×N attention score matrix in HBM. The speedup and memory savings of FlashAttention are the same phenomenon you observe in this lab, applied to the attention operation. Once you have felt the HBM-vs-SRAM tradeoff at the kernel level, every kernel optimization in the LLM stack (fused kernels, quantized kernels, FlashDecoding) makes intuitive sense.

## Variant Experiments

- Measure the effect of tile size T on performance — there is a sweet spot where SRAM fills but isn't exceeded; beyond that, occupancy drops.
- Compare your tiled kernel to `torch.matmul` (which uses cuBLAS or cuDNN under the hood) — the gap shows how much work goes into a production-grade kernel.
- Try a version without `__syncthreads()` and observe the race-condition artifacts — a concrete demonstration of why barrier synchronization is required.
