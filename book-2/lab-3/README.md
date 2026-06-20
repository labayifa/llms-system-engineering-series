# Lab 3 — Run Data-Parallel Training

**Book 2 · Part II (distributed training, DDP, FSDP)**

---

## Hypothesis

Data parallelism (DDP) scales training throughput nearly linearly with the number of GPUs until communication overhead bites. FSDP achieves similar throughput but with significantly lower peak memory per GPU by sharding model weights, gradients, and optimizer state.

## Background

**DDP (DistributedDataParallel):** each GPU holds a full replica of the model. After each backward pass, gradients are all-reduced across all GPUs (ring-allreduce), then each replica updates its own copy. Fast but memory-heavy — every GPU holds the full model + gradients + optimizer state.

**FSDP (Fully Sharded Data Parallelism / ZeRO Stage 3):** model parameters, gradients, and optimizer state are sharded across GPUs. Before each forward/backward step, each GPU issues an all-gather to reconstruct the full layer it needs, computes its shard, then discards the gathered parameters (reduce-scatter in backward). Memory per GPU drops roughly by the number of GPUs, at the cost of more communication.

## Setup

```bash
pip install torch torchvision  # torchrun ships with torch

# Confirm torchrun is available
torchrun --version
```

You need at least 2 GPUs for a meaningful comparison. Single-GPU "distributed" runs with `nproc_per_node=1` are valid for functional testing but won't show the scaling.

## Procedure

### Part A — DDP throughput scaling
1. Write a training loop (`train.py`) that trains a small transformer (or any model that fits in GPU memory ×1) on synthetic data for 100 steps, printing tokens/sec at the end.
2. Run with N = 1, 2, and 4 GPUs using DDP:
   ```bash
   torchrun --nproc_per_node=N train.py --strategy ddp
   ```
3. Record tokens/sec for each N. Plot throughput vs N and note where it deviates from linear.

### Part B — FSDP memory reduction
1. Repeat the same training with FSDP wrapping:
   ```bash
   torchrun --nproc_per_node=N train.py --strategy fsdp
   ```
2. Record peak VRAM per GPU with `nvidia-smi` or `torch.cuda.max_memory_allocated()`.
3. Compare peak memory per GPU: DDP vs FSDP at N=4.

## Expected Observations

- **DDP throughput scales near-linearly at small N** (N=1→2 and N=2→4), then deviates as the allreduce communication time becomes a larger fraction of the step time.
- **FSDP peak memory per GPU is substantially lower** than DDP — roughly `(full model size) / N` for the parameters, plus activation memory per GPU.
- FSDP may show slightly lower throughput than DDP at the same N due to the additional all-gather / reduce-scatter communication pattern.
- At N=1, DDP and FSDP should have identical throughput (no distributed communication).

## Why This Matters

Choosing between DDP and FSDP is one of the first decisions when scaling training. The rule of thumb: use DDP when your model fits comfortably in GPU memory (fast, simple, minimal communication overhead). Switch to FSDP when the model approaches or exceeds per-GPU memory — trading communication for the ability to train at all. This lab makes that tradeoff concrete rather than theoretical, and gives you the numbers to make the decision for your own hardware.

## Variant Experiments

- Benchmark FSDP with different `ShardingStrategy` values (`FULL_SHARD` vs `SHARD_GRAD_OP` vs `NO_SHARD`) and observe the memory-throughput tradeoff.
- Add gradient checkpointing (`torch.utils.checkpoint`) to the DDP run and measure the memory reduction vs the compute-time penalty (recomputing activations in the backward pass).
- Profile the DDP allreduce time with `torch.profiler` and plot it as a fraction of total step time as N grows — the point where communication dominates is your scaling ceiling.
